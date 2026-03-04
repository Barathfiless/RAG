from flask import Flask, render_template, request, jsonify, Response
import os
import json
import time
import shutil
import threading
import database as db
import faiss

from utils import (
    extract_content, chunk_content, build_index, search,
    synthesize_answer, generate_suggestions,
    detect_page_query, get_page_content, generate_followup_suggestions
)
from features import is_image_query, generate_quiz, generate_outline, generate_glossary

_ocr_reader = None
def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        import torch
        # use GPU if available
        gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(['en'], gpu=gpu)
    return _ocr_reader

print("Starting Flask app...")
app = Flask(__name__)
print("Initializing database...")
db.init_db()
print("Database initialized.")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Cache: {session_id: {index, chunks, stats, suggestions, page_map}}
vector_cache = {}

# Processing Status: {session_id: {status: "loading|processing|ready|error", message: "..."}}
processing_status = {}


# ─── Background Processing ────────────────────────────────────────────────────

def process_pdf_background(session_id, filepath, filename):
    try:
        processing_status[session_id] = {"status": "processing", "message": "Extracting text and images..."}
        
        # 1. Extraction (Fast)
        content_list, stats, page_map = extract_content(filepath, str(session_id))
        
        processing_status[session_id] = {"status": "processing", "message": "Chunking content..."}
        # 2. Chunking
        chunks = chunk_content(content_list)
        
        processing_status[session_id] = {"status": "processing", "message": "Building vector index..."}
        # 3. Indexing (Slowest)
        index = build_index(chunks)
        
        processing_status[session_id] = {"status": "processing", "message": "Generating initial suggestions..."}
        # 4. Suggestions
        suggestions = generate_suggestions(chunks)
        
        # 5. Finalize images - no longer needed as they are already in the correct folder
        # but we still need to fix the URLs if extract_content uses relative paths

        # 6. Cache
        vector_cache[session_id] = {
            "chunks":      chunks,
            "index":       index,
            "stats":       stats,
            "page_map":    page_map,
            "suggestions": suggestions
        }
        
        # 7. Update DB
        db.update_session_stats(session_id, stats["pages"], len(chunks), stats["images"])
        
        welcome = (f"**{filename}** indexed successfully.\n\n"
                   f"Pages: **{stats['pages']}** · "
                   f"Searchable chunks: **{len(chunks)}** · "
                   f"Images extracted: **{stats['images']}**\n\n"
                   f"Ask me anything, or try asking: *\"Show page 3\"* to view any page directly.")
        db.add_message(session_id, "bot", welcome)
        
        processing_status[session_id] = {
            "status": "ready", 
            "stats": {**stats, "chunks": len(chunks)},
            "suggestions": suggestions,
            "filename": filename
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        processing_status[session_id] = {"status": "error", "message": str(e)}

# ─── Lazy-load vector store ──────────────────────────────────────────────────

def ensure_vector_store(session_id):
    if session_id in vector_cache:
        return True

    conn    = db.get_db_connection()
    session = conn.execute('SELECT * FROM sessions WHERE id = ?', (session_id,)).fetchone()
    conn.close()

    if session and session['filename']:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], session['filename'])
        if os.path.exists(filepath):
            try:
                content_list, stats, page_map = extract_content(filepath, session_id)
                chunks      = chunk_content(content_list)
                index       = build_index(chunks)
                suggestions = generate_suggestions(chunks)
                vector_cache[session_id] = {
                    "chunks":      chunks,
                    "index":       index,
                    "stats":       stats,
                    "page_map":    page_map,
                    "suggestions": suggestions
                }
                return True
            except Exception as e:
                print(f"[Lazy load error] Session {session_id}: {e}")
                return False
    return False


# ─── Favicon & Status ────────────────────────────────────────────────────────

@app.route('/favicon.ico')
def favicon():
    return Response(status=204)

@app.route('/process_status/<int:session_id>', methods=['GET'])
def get_process_status(session_id):
    status = processing_status.get(session_id, {"status": "not_found"})
    return jsonify(status)


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        session_id = request.form.get('session_id')

        # ── PDF Upload ──────────────────────────────────────────────────
        if "pdf" in request.files:
            file = request.files["pdf"]
            if not file.filename:
                return jsonify({"error": "No file selected."}), 400

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Create session skeleton first
            new_session_id = db.create_session(title=file.filename, filename=file.filename)
            
            # Start background processing
            thread = threading.Thread(target=process_pdf_background, args=(new_session_id, filepath, file.filename))
            thread.daemon = True
            thread.start()
            
            return jsonify({
                "status": "processing",
                "session_id": new_session_id,
                "filename": file.filename
            })

        # ── Question ─────────────────────────────────────────────────────
        question = request.form.get("question", "").strip()
        if question and session_id:
            session_id = int(session_id)

            if not ensure_vector_store(session_id):
                return jsonify({"error": "Could not load PDF. Please upload again."}), 400

            cache = vector_cache[session_id]
            stats = cache.get("stats", {})

            try:
                # ── STEP 1: Check for page-specific query ─────────────
                page_num = detect_page_query(question)

                if page_num is not None:
                    total_pages = stats.get("pages", 0)
                    page_text, page_images, err = get_page_content(
                        page_num, cache["page_map"], total_pages
                    )

                    if err:
                        answer = err
                        pages  = []
                        images = []
                    else:
                        # Split raw page text into readable ~4-sentence paragraphs
                        from nltk.tokenize import sent_tokenize as _stok
                        sents  = _stok(page_text)
                        groups, buf = [], []
                        for s in sents:
                            buf.append(s.strip())
                            if len(buf) >= 4:
                                groups.append(" ".join(buf))
                                buf = []
                        if buf:
                            groups.append(" ".join(buf))
                        body   = "\n\n".join(groups)
                        answer = f"## Page {page_num}\n\n---\n\n{body}"
                        pages  = [page_num]
                        images = page_images

                    # Follow-up suggestions seeded from the raw page text
                    followups = generate_followup_suggestions(
                        question,
                        page_text if not err else "",
                        cache["chunks"], cache["index"], n=4
                    )

                    db.add_message(session_id, "user", question)
                    db.add_message(session_id, "bot", answer,
                                   metadata={"pages": pages, "images": images})

                    return jsonify({
                        "question":    question,
                        "answer":      answer,
                        "pages":       pages,
                        "images":      images,
                        "suggestions": followups,
                        "mode":        "page_lookup",
                        "status":      "success"
                    })

                # ── STEP 2: Semantic search ────────────────────────────
                conv_context = db.get_recent_messages(session_id, n=6)
                results = search(
                    question,
                    cache["index"],
                    cache["chunks"],
                    top_k=5,
                    conversation_context=conv_context
                )

                if not results:
                    return jsonify({
                        "question": question,
                        "answer":   "No relevant content found. Try rephrasing your question.",
                        "pages":    [],
                        "images":   [],
                        "mode":     "semantic",
                        "status":   "success"
                    })

                info = db.get_session_info(session_id) or {"title": "Unknown PDF"}
                answer, pages = synthesize_answer(question, results, stats=stats, filename=info.get('title'), conversation_context=conv_context)
                # Only include images if the query explicitly asks for visual content
                if is_image_query(question):
                    images = list(dict.fromkeys(
                        img for r in results for img in r.get("images", [])
                    ))
                else:
                    images = []

                # Generate context-aware follow-up suggestions
                followups = generate_followup_suggestions(
                    question, answer,
                    cache["chunks"], cache["index"], n=4
                )

                db.add_message(session_id, "user", question)
                db.add_message(session_id, "bot", answer,
                               metadata={"pages": pages, "images": images})

                return jsonify({
                    "question":    question,
                    "answer":      answer,
                    "pages":       pages,
                    "images":      images,
                    "suggestions": followups,
                    "mode":        "semantic",
                    "status":      "success"
                })

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                return jsonify({"error": f"Search failed: {str(e)}"}), 500

    history = db.get_all_sessions()
    return render_template("index.html", history=history)


@app.route("/get_messages/<int:session_id>", methods=["GET"])
def get_messages(session_id):
    messages = db.get_session_messages(session_id)
    return jsonify({"messages": messages})


@app.route("/session_info/<int:session_id>", methods=["GET"])
def session_info(session_id):
    info = db.get_session_info(session_id)
    if not info:
        return jsonify({"error": "Session not found"}), 404

    suggestions = []
    if session_id in vector_cache:
        suggestions = vector_cache[session_id].get("suggestions", [])
    elif ensure_vector_store(session_id):
        suggestions = vector_cache[session_id].get("suggestions", [])

    return jsonify({**info, "suggestions": suggestions})


@app.route("/export/<int:session_id>", methods=["GET"])
def export_chat(session_id):
    info     = db.get_session_info(session_id)
    messages = db.get_session_messages(session_id)
    if not info:
        return jsonify({"error": "Session not found"}), 404

    lines = [
        "AI PDF Assistant — Chat Export",
        f"Document : {info['title']}",
        f"Exported : {time.strftime('%Y-%m-%d %H:%M')}",
        "=" * 60, ""
    ]
    for m in messages:
        prefix = "You" if m["sender"] == "user" else "AI"
        lines.append(f"[{prefix}]\n{m['content']}\n")

    return Response(
        "\n".join(lines),
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename=chat_{session_id}.txt"}
    )


@app.route("/rename_session", methods=["POST"])
def rename_session():
    session_id = request.form.get("session_id")
    title      = request.form.get("title", "").strip()
    if not session_id or not title:
        return jsonify({"error": "Missing parameters"}), 400
    db.rename_session(int(session_id), title)
    return jsonify({"status": "success"})


@app.route("/delete_session", methods=["POST"])
def delete_session():
    session_id = request.form.get('session_id')
    if session_id:
        sid = int(session_id)
        db.delete_session(sid)
        vector_cache.pop(sid, None)
        img_dir = os.path.join('static', 'extracted_images', str(sid))
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir, ignore_errors=True)
        return jsonify({"status": "success"})
    return jsonify({"error": "No session ID provided"}), 400


@app.route("/ask_with_image", methods=["POST"])
def ask_with_image():
    """
    Accept an image (and optional text question) from the user.
    1. Try OCR via pytesseract to extract text from the image
    2. Combine OCR text + user question as the search query
    3. Run the same semantic search pipeline and return an answer
    """
    session_id = request.form.get("session_id")
    question   = request.form.get("question", "").strip()
    img_file   = request.files.get("image")

    if not session_id or not img_file:
        return jsonify({"error": "session_id and image are required"}), 400

    session_id = int(session_id)
    if not ensure_vector_store(session_id):
        return jsonify({"error": "Could not load PDF. Please upload again."}), 400

    cache = vector_cache[session_id]

    # ── OCR the uploaded image ───────────────────────────────────────────────
    ocr_text = ""
    ocr_note = ""
    try:
        reader = get_ocr_reader()
        img_bytes = img_file.read()
        img_file.seek(0)
        
        result = reader.readtext(img_bytes, detail=0)
        ocr_text = " ".join(result).strip()
        
        if ocr_text:
            ocr_note = f"**Text extracted from your image:**\n> {ocr_text[:400]}{'...' if len(ocr_text) > 400 else ''}\n\n---\n\n"
    except Exception as e:
        # Fallback without sending an ugly error to the user
        ocr_note = ""

    # Save uploaded image to static for display in chat
    img_file.seek(0)
    upload_img_dir = os.path.join('static', 'uploaded_images')
    os.makedirs(upload_img_dir, exist_ok=True)
    img_ext      = img_file.filename.rsplit('.', 1)[-1].lower() if '.' in img_file.filename else 'png'
    img_filename = f"user_{session_id}_{int(time.time())}.{img_ext}"
    img_save_path = os.path.join(upload_img_dir, img_filename)
    img_file.seek(0)
    img_file.save(img_save_path)
    user_img_url = f"/static/uploaded_images/{img_filename}"

    # ── Build combined query ─────────────────────────────────────────────────
    parts = [p for p in [question, ocr_text] if p]
    full_query = " ".join(parts) if parts else "Describe this image"

    try:
        # Check for explicit page query first
        page_num = detect_page_query(full_query)
        if page_num is not None:
            stats = cache.get("stats", {})
            page_text, page_images, err = get_page_content(
                page_num, cache["page_map"], stats.get("pages", 0)
            )
            if err:
                answer = ocr_note + err
            else:
                answer = ocr_note + f"## Page {page_num}\n\n{page_text}"
            pages  = [] if err else [page_num]
            images = page_images
        else:
            conv_context = db.get_recent_messages(session_id, n=6)
            results      = search(full_query, cache["index"], cache["chunks"],
                                  top_k=5, conversation_context=conv_context)
            if results:
                info = db.get_session_info(session_id) or {"title": "Unknown PDF"}
                ans, pages = synthesize_answer(full_query, results, stats=cache.get("stats", {}), filename=info.get('title'), conversation_context=conv_context)
                answer     = ocr_note + ans
                images     = list(dict.fromkeys(
                    img for r in results for img in r.get("images", [])
                ))
            else:
                answer = ocr_note + "No relevant content found in the PDF for this image."
                pages, images = [], []

        followups = generate_followup_suggestions(
            full_query, answer, cache["chunks"], cache["index"], n=4
        )

        db.add_message(session_id, "user",
                       question or "(Image uploaded)",
                       metadata={"images": [user_img_url]})
        db.add_message(session_id, "bot", answer,
                       metadata={"pages": pages, "images": images})

        return jsonify({
            "answer":      answer,
            "pages":       pages,
            "images":      images,
            "user_image":  user_img_url,
            "ocr_text":    ocr_text,
            "suggestions": followups,
            "status":      "success"
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500



# ─── Advanced Feature Routes ───────────────────────────────────────────

@app.route('/quiz/<int:session_id>', methods=['GET'])
def quiz(session_id):
    """Generate MCQ quiz questions from the PDF content."""
    n = int(request.args.get('n', 5))
    if not ensure_vector_store(session_id):
        return jsonify({"error": "Session not found or PDF not indexed."}), 404
    cache = vector_cache[session_id]
    questions = generate_quiz(cache['chunks'], n=n)
    return jsonify({"questions": questions, "status": "success"})


@app.route('/outline/<int:session_id>', methods=['GET'])
def outline(session_id):
    """Generate a structured outline/table-of-contents for the PDF."""
    if not ensure_vector_store(session_id):
        return jsonify({"error": "Session not found or PDF not indexed."}), 404
    cache = vector_cache[session_id]
    toc = generate_outline(cache['chunks'])
    return jsonify({"outline": toc, "status": "success"})


@app.route('/glossary/<int:session_id>', methods=['GET'])
def glossary(session_id):
    """Auto-generate a glossary of key terms from the PDF."""
    if not ensure_vector_store(session_id):
        return jsonify({"error": "Session not found or PDF not indexed."}), 404
    cache = vector_cache[session_id]
    terms = generate_glossary(cache['chunks'])
    return jsonify({"glossary": terms, "status": "success"})


if __name__ == "__main__":
    # use_reloader=False prevents the app from starting twice, saving memory and CPU
    app.run(debug=True, port=5000, use_reloader=False)
