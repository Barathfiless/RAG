import faiss
import re
import fitz
import os
import random
import numpy as np
import torch
import multiprocessing
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import sent_tokenize

# Set torch threads globally for better performance on CPUs
if not torch.cuda.is_available():
    try:
        cpu_count = multiprocessing.cpu_count()
        torch.set_num_threads(max(1, int(cpu_count * 0.8)))
    except Exception:
        pass

# ─── Configuration & Constants ───────────────────────────────────────────────

_STOP = {
    'the','a','an','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could',
    'should','may','might','shall','can','need','dare','ought',
    'in','on','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below',
    'from','up','down','out','off','over','under','again','further',
    'then','once','this','that','these','those','it','its','they',
    'them','their','we','our','you','your','he','she','his','her',
    'and','or','but','if','while','although','because','since',
    'as','so','yet','both','either','neither','not','no','nor',
    'just','also','each','all','any','some','such','same','other',
    'more','most','much','many','few','little','own','very','just',
    'of','to'
}

_GENERIC_PDF_WORDS = {
    'application', 'applications', 'introduction', 'summary', 'conclusion',
    'chapter', 'section', 'page', 'table', 'figure', 'content', 'contents',
    'index', 'references', 'example', 'exercises', 'questions', 'appendix',
    'background', 'overview', 'basic', 'basics', 'advanced', 'study', 'notes',
    'lecture', 'module', 'unit', 'part', 'description', 'definition', 'scope',
    'preface', 'acknowledgments', 'abstract', 'objective', 'objectives',
    'according', 'within', 'between', 'artificial', 'intelligence'
}

_nltk_loaded = False
def _ensure_nltk():
    global _nltk_loaded
    if not _nltk_loaded:
        for resource in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}...")
                try:
                    nltk.download(resource, quiet=True)
                    print(f"NLTK resource {resource} downloaded.")
                except Exception as e:
                    print(f"Failed to download NLTK resource {resource}: {e}")
        _nltk_loaded = True

_model = None

def _get_model():
    global _model
    if _model is None:
        _ensure_nltk()
        print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        print(f"SentenceTransformer model loaded successfully on {device}.")
    return _model

# Cross-encoder model for more accurate re-ranking of retrieved chunks.
_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None


def _get_cross_encoder():
    """
    Lazy-load and cache the cross-encoder model using the detected device.
    """
    global _cross_encoder
    if _cross_encoder is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL_NAME, device=device)
        except Exception:
            _cross_encoder = False
    return _cross_encoder if _cross_encoder is not False else None


# ─── Emoji Stripping ──────────────────────────────────────────────────────────

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE
)

def strip_emojis(text: str) -> str:
    return _EMOJI_RE.sub('', text).strip()


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = _EMOJI_RE.sub('', text) # Inline strip_emojis for speed
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Preservation tweak: Keep double newlines to maintain paragraph/header structure
    # Only collapse single newlines (usually line wraps in PDFs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = re.sub(r'[\uE000-\uF8FF]', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\xad]', '', text)
    # Basic space normalization but keep the newlines we decided to keep
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_content(pdf_path: str, session_id):
    """
    Returns (content_list, stats, page_map).
    page_map: {page_num: {"text": str, "images": [url]}}
    """
    # Use 'threading' or 'process' safe opening if needed, but fitz is fast.
    doc      = fitz.open(pdf_path)
    content  = []
    stats    = {"pages": len(doc), "images": 0}
    page_map = {}

    img_dir = os.path.join('static', 'extracted_images', str(session_id))
    os.makedirs(img_dir, exist_ok=True)

    for page_num, page in enumerate(doc):
        pn = page_num + 1

        # Optimization: use 'text' flag for faster extraction if tables/formatting aren't critical
        raw     = page.get_text("text") 
        cleaned = clean_text(raw)
        
        if cleaned and len(cleaned) > 20: # Slightly lower threshold to not miss short meaningful pages
            content.append({"type": "text", "content": cleaned, "page": pn})
            page_map.setdefault(pn, {"text": "", "images": []})
            page_map[pn]["text"] = cleaned

        # Optimization: Limit images per page to avoid getting stuck on image-heavy documents
        page_images = page.get_images(full=True)
        if len(page_images) > 10: 
            page_images = page_images[:10]

        for img_index, img in enumerate(page_images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]
                ext        = base_image["ext"]
                
                # Filter out tiny images (icons, spacers, bullets) - increased to 10KB
                if len(img_bytes) < 10240: 
                    continue
                    
                fname    = f"page{pn}_img{img_index}.{ext}"
                img_path = os.path.join(img_dir, fname)
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                    
                url = f"/static/extracted_images/{session_id}/{fname}"
                content.append({"type": "image", "url": url, "page": pn})
                page_map.setdefault(pn, {"text": "", "images": []})
                page_map[pn]["images"].append(url)
                stats["images"] += 1
            except Exception:
                continue

    doc.close()
    return content, stats, page_map



# ─── Page Query Detection ─────────────────────────────────────────────────────

_ORDINALS = {
    "first": 1,  "second": 2,  "third": 3,  "fourth": 4,  "fifth": 5,
    "sixth": 6,  "seventh": 7, "eighth": 8, "ninth": 9,  "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}

# All patterns that unambiguously reference a specific page number
_PAGE_PATTERNS = [
    # "page 3", "page number 3", "page no 3", "page no. 3", "page #3"
    re.compile(r'\bpage\s*(?:number|num|no\.?|#)?\s*(\d+)\b', re.I),

    # "go to page 3", "open page 3", "jump to page 3", "navigate to page 3"
    re.compile(r'\b(?:go\s+to|open|jump\s+to|navigate\s+to|load|scroll\s+to)\s+page\s+(\d+)\b', re.I),

    # "show me page 3", "show page 3", "display page 3", "give page 3"
    re.compile(r'\b(?:show(?:\s+me)?|display|give(?:\s+me)?|print|fetch|get)\s+page\s+(\d+)\b', re.I),

    # "what is on page 3", "what's on page 3", "whats on page 3"
    re.compile(r"\bwhat(?:'?s|is)\s+(?:on\s+)?page\s+(\d+)\b", re.I),

    # "contents of page 3", "content of page 3", "text of page 3"
    re.compile(r'\b(?:content|contents|text|info|information)\s+(?:of|on|from|in)\s+page\s+(\d+)\b', re.I),

    # "page 3 content", "page 3 details", "page 3 text"
    re.compile(r'\bpage\s+(\d+)\s+(?:content|contents|text|details|info|summary)', re.I),

    # "tell me about page 3", "tell me what's on page 3"
    re.compile(r'\btell\s+(?:me\s+)?(?:about\s+)?(?:what(?:\'?s\s+on\s+)?)page\s+(\d+)\b', re.I),

    # "read page 3", "see page 3", "view page 3", "check page 3"
    re.compile(r'\b(?:read|see|view|check|look\s+at|describe)\s+page\s+(\d+)\b', re.I),

    # "3rd page", "2nd page", "1st page", "21st page" etc.
    re.compile(r'\b(\d+)(?:st|nd|rd|th)\s+page\b', re.I),

    # "pg 3", "pg. 3"
    re.compile(r'\bpg\.?\s*(\d+)\b', re.I),

    # "p.3", "p. 3" — only if isolated (not inside a word like "app.3")
    re.compile(r'(?<!\w)p\.\s*(\d+)\b', re.I),
]

def detect_page_query(query: str):
    """
    Returns page_number (int) if the query unambiguously asks for a specific page.
    Returns None if no page-specific intent is detected.
    """
    q = query.strip()

    # Try all numeric patterns first
    for pat in _PAGE_PATTERNS:
        m = pat.search(q)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                pass

    # Try ordinal words: "third page", "second page"
    for word, num in _ORDINALS.items():
        if re.search(rf'\b{word}\s+page\b', q, re.I):
            return num

    return None


def handle_meta_query(query: str, stats: dict, filename: str):
    """
    Detects and answers questions about the PDF document itself (meta-questions).
    Returns (answer, pages_list) or (None, [])
    """
    q = query.lower().strip()
    
    # Page count
    if any(p in q for p in ["how many pages", "total pages", "number of pages", "length of the pdf", "how long is this"]):
        count = stats.get('pages', 0)
        return f"This document has **{count}** pages in total.", []
        
    # Image count
    if any(p in q for p in ["how many images", "total images", "extracted images", "number of images", "any pictures"]):
        count = stats.get('images', 0)
        if count == 0:
            return "No images were found or extracted from this document.", []
        return f"I have extracted **{count}** images from this document.", []
        
    # Filename / Identity
    if any(p in q for p in ["what is the name", "filename", "name of this document", "which pdf is this", "identify this file"]):
        name = filename or "the uploaded PDF"
        return f"The current file is named **{name}**.", []

    # Summary of structure
    if any(p in q for p in ["what can you do", "help me", "how to use"]):
        return (f"I've read **{filename}** ({stats.get('pages', 0)} pages). "
                "You can ask me about its content, specify pages (e.g., 'show page 5'), "
                "or search for specific topics mentioned in the text."), []
        
    return None, []


def get_page_content(page_num: int, page_map: dict, total_pages: int):
    """
    Returns (text, images, error_msg).
    text is the clean, raw page text stripped of leading page-number artifacts.
    error_msg is None on success.
    """
    if page_num < 1 or page_num > total_pages:
        return None, [], f"This document only has {total_pages} pages."

    data = page_map.get(page_num)
    if not data or not data.get("text"):
        return None, [], f"Page {page_num} appears to be blank or contains only images."

    text = data["text"].strip()

    # Strip leading page number artifacts that PDF extraction sometimes includes
    # e.g. "3 Module 1 ARTIFICIAL INTELLIGENCE…" — remove leading digit if it matches page num
    text = re.sub(rf'^{page_num}\s+', '', text).strip()

    return text, data.get("images", []), None




# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_content(content_list: list) -> list:
    """
    Chunks text into precise, fine-grained pieces for maximum retrieval accuracy.
    GOD-MODE: Smaller WINDOW=5 sentences ensures the right section is always retrieved.
    Larger overlaps (2 sentences) ensure no context is lost at boundaries.
    """
    _ensure_nltk()
    # GOD-MODE: Smaller window = more precise retrieval
    # A 5-sentence chunk (~150 words) maps to a tight, specific concept
    WINDOW  = 5
    OVERLAP = 2

    page_images = {}
    for item in content_list:
        if item["type"] == "image":
            page_images.setdefault(item["page"], []).append(item["url"])

    final_chunks = []
    chunk_idx    = 0

    for item in content_list:
        if item["type"] != "text":
            continue
        # Split on newlines first to keep section headers intact
        paragraphs = [p.strip() for p in item["content"].split('\n\n') if p.strip()]
        page         = item["page"]
        assoc_images = page_images.get(page, [])

        for para in paragraphs:
            sentences = sent_tokenize(para)
            if len(sentences) <= WINDOW:
                final_chunks.append({
                    "text": para,
                    "images": assoc_images,
                    "page": page,
                    "chunk_index": chunk_idx
                })
                chunk_idx += 1
            else:
                step = WINDOW - OVERLAP
                for i in range(0, len(sentences), step):
                    batch = sentences[i: i + WINDOW]
                    if not batch:
                        break
                    final_chunks.append({
                        "text": " ".join(batch),
                        "images": assoc_images,
                        "page": page,
                        "chunk_index": chunk_idx
                    })
                    chunk_idx += 1
                    if i + WINDOW >= len(sentences):
                        break

    return final_chunks


# ─── Indexing ─────────────────────────────────────────────────────────────────

def build_index(chunks: list):
    """
    Builds a FAISS index from text chunks.
    Optimization: Direct tensor conversion and normalized embeddings.
    """
    if not chunks:
        return None
    
    texts = [c["text"] for c in chunks]
    model = _get_model()
    
    # Use normalize_embeddings=True for cosine similarity with IndexFlatIP
    embs = model.encode(
        texts, 
        batch_size=128,
        show_progress_bar=False, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
        
    embs  = np.array(embs).astype('float32')
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


# ─── BM25 Scoring ─────────────────────────────────────────────────────────────

def _bm25_score(query_tokens: list, doc_text: str, k1=1.5, b=0.75, avg_len=100) -> float:
    doc_tokens = doc_text.lower().split()
    doc_len    = len(doc_tokens)
    tf_map     = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        tf = tf_map.get(qt, 0)
        if tf:
            tf_n   = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += 1.0 * tf_n
    return score


# ─── MMR Re-ranking ───────────────────────────────────────────────────────────

def _mmr(query_emb, cand_embs, cand_indices, top_k=4, lam=0.65):
    selected  = []
    remaining = list(range(len(cand_indices)))
    relevance = np.dot(cand_embs, query_emb).flatten()
    for _ in range(min(top_k, len(remaining))):
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_embs = cand_embs[selected]
            best = max(
                remaining,
                key=lambda i: lam * relevance[i] - (1 - lam) * float(np.max(np.dot(sel_embs, cand_embs[i])))
            )
        selected.append(best)
        remaining.remove(best)
    return [cand_indices[i] for i in selected]


def _correct_query_spelling(query: str, chunks: list) -> str:
    """
    Very lightweight document-aware 'spell correction'.
    Compares query words against the document's own vocabulary.
    """
    import difflib
    
    # 1. Build a small doc-specific vocabulary from chunks
    # (In a real app, this would be cached in the session cache)
    vocab = set()
    for c in chunks[:100]:
        words = re.findall(r'\b[A-Za-z]{4,}\b', c["text"])
        for w in words:
            vocab.add(w.lower())
    
    words = query.split()
    corrected = []
    for w in words:
        clean_w = re.sub(r'[^a-zA-Z]', '', w).lower()
        if len(clean_w) < 4:
            corrected.append(w)
            continue
            
        # If the word is already likely 'correct' (in vocab), leave it
        if clean_w in vocab or clean_w in _STOP or clean_w in _GENERIC_PDF_WORDS:
            corrected.append(w)
            continue
            
        # Try to find a close match in the document's vocabulary
        matches = difflib.get_close_matches(clean_w, list(vocab), n=1, cutoff=0.8)
        if matches:
            # Preserve original casing if possible, but use the doc's word
            corrected.append(matches[0])
        else:
            corrected.append(w)
            
    return " ".join(corrected)


# ─── Query Expansion ──────────────────────────────────────────────────────────

def expand_query(query: str, chunks: list = None, conversation_context=None) -> str:
    query    = query.strip()
    
    # Apply doc-aware spell correction if chunks are provided
    if chunks:
        query = _correct_query_spelling(query, chunks)
        
    expanded = query
    if query.endswith('?'):
        expanded += ' ' + query.rstrip('?').strip()
    if conversation_context:
        follow_ups = {'it', 'this', 'that', 'they', 'them', 'he', 'she', 'explain', 'more', 'detail', 'elaborate'}
        if set(query.lower().split()) & follow_ups:
            last_bot = next(
                (m['content'] for m in reversed(conversation_context) if m['sender'] == 'bot'),
                None
            )
            if last_bot:
                ctx_words = [w for w in last_bot.split() if len(w) > 4 and w.isalpha()][:10]
                expanded += ' ' + ' '.join(ctx_words)
    return expanded


# ─── Semantic Search ──────────────────────────────────────────────────────────

def search(query: str, index, chunks: list, top_k=5, conversation_context=None) -> list:
    """GOD-MODE Search: Large pool, always-on cross-encoder, exact-phrase boosting."""
    if index is None or not chunks:
        return []

    expanded  = expand_query(query, chunks, conversation_context)
    q_emb     = _get_model().encode([expanded], normalize_embeddings=True, show_progress_bar=False)
    q_emb     = np.array(q_emb).astype('float32')
    q_tokens  = [w.lower() for w in query.split() if len(w) > 2]
    q_lower   = query.lower()
    ex_lower  = expanded.lower()

    # GOD-MODE: Pull 8x more candidates for a deeper rerank pool
    k_cands = min(max(top_k * 8, 40), len(chunks))
    scores, indices = index.search(q_emb, k=k_cands)

    valid_idx     = [int(i) for i in indices[0] if i != -1]
    cosine_scores = {
        int(indices[0][j]): float(scores[0][j])
        for j in range(len(indices[0])) if indices[0][j] != -1
    }
    if not valid_idx:
        return []

    avg_len = sum(len(chunks[i]["text"].split()) for i in valid_idx) / max(len(valid_idx), 1)
    hybrid  = []

    # Detect section numbers/headers in query: "2.5", "Module 1", "3.1.2"
    section_nums = re.findall(r'\b\d+(?:\.\d+)+\b', expanded)
    # Detect bold/title phrases: query is often the exact header text
    query_phrases = [q_lower, ex_lower]

    for i in valid_idx:
        chunk_text = chunks[i]["text"]
        chunk_lower = chunk_text.lower()
        bm = _bm25_score(q_tokens, chunk_text, avg_len=avg_len)
        cs = cosine_scores.get(i, 0.0)

        boost = 1.0
        # Boost 1: chunk contains exact section number from query (e.g., "2.5")
        for sec in section_nums:
            if sec in chunk_text:
                boost += 1.0  # Very strong boost for exact section match

        # Boost 2: chunk contains the entire query as a substring (header match)
        for phrase in query_phrases:
            if phrase in chunk_lower:
                boost += 0.75
                break

        # Boost 3: high token overlap (all main query words present)
        tokens_present = sum(1 for t in q_tokens if t in chunk_lower)
        if q_tokens and tokens_present / len(q_tokens) > 0.8:
            boost += 0.3

        score = (0.65 * cs + 0.35 * (bm / (bm + 1))) * boost
        hybrid.append((i, score))

    hybrid.sort(key=lambda x: x[1], reverse=True)

    # GOD-MODE: Always run cross-encoder on top-15 candidates for maximum precision
    top_hybrid = [i for i, _ in hybrid[:15]]
    ce_model = _get_cross_encoder()
    if ce_model is not None and top_hybrid:
        try:
            pairs = [(expanded, chunks[i]["text"][:2000]) for i in top_hybrid]
            ce_scores = ce_model.predict(pairs)
            ranked_pairs = sorted(
                zip(top_hybrid, ce_scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            top_hybrid = [idx for idx, _ in ranked_pairs]
        except Exception:
            pass

    # Trim to our final pool
    top_hybrid = top_hybrid[:int(top_k * 2)]

    try:
        cand_embs = np.array([index.reconstruct(int(i)) for i in top_hybrid]).astype('float32')
    except Exception:
        cand_texts = [chunks[i]["text"] for i in top_hybrid]
        cand_embs  = _get_model().encode(cand_texts, normalize_embeddings=True, show_progress_bar=False)
        cand_embs  = np.array(cand_embs).astype('float32')

    # MMR: ensure diversity but keep top_k results
    ranked = _mmr(q_emb[0], cand_embs, top_hybrid, top_k=top_k)

    # Attach hybrid boost scores for use downstream in synthesize
    hybrid_map = {i: s for i, s in hybrid}
    results = []
    for i in ranked:
        results.append({
            "text":        chunks[i]["text"],
            "images":      chunks[i].get("images", []),
            "page":        chunks[i].get("page", "?"),
            "chunk_index": chunks[i].get("chunk_index", i),
            "_score":      hybrid_map.get(i, 0.0),
        })
    return results



# ─── Answer Synthesis ─────────────────────────────────────────────────────────

def synthesize_answer(question: str, results: list, stats: dict = None, filename: str = None, conversation_context=None):
    """
    GOD-MODE Answer Synthesis:
    1. Check for Meta-Questions (Page counts, stats)
    2. Pick the highest-confidence result as the 'anchor'
    3. If adjacent chunks exist on same page, stitch them together for completeness
    4. Return verbatim PDF text (preserving ALL layout, newlines, structure)
    Returns (answer_markdown: str, sorted_pages: list)
    """
    # Intent 1: Meta-Questions (Filename, Pages, etc.)
    if stats:
        meta_ans, _ = handle_meta_query(question, stats, filename)
        if meta_ans:
            return meta_ans, []

    if not results:
        return "I could not find relevant information for that question. Try rephrasing.", []

    # Sort all results by internal score (cross-encoder + boost), best first
    results_sorted = sorted(results, key=lambda r: r.get("_score", 0), reverse=True)

    # Take the best result as the anchor chunk
    best = results_sorted[0]
    best_page   = best.get("page", 0)
    best_ci     = best.get("chunk_index", 0)

    # GOD-MODE Adjacency Stitching:
    # Find chunks from the same page that are adjacent (neighboring sections)
    # and concatenate them so the full section content is returned.
    all_pages = []
    seen_ci   = {best_ci}
    parts     = [best["text"].strip()]
    all_images = list(best.get("images", []))

    if best_page not in all_pages:
        all_pages.append(best_page)

    # Look for chunks immediately adjacent (ci ± 1, ± 2) on the SAME page
    for r in results:
        if r.get("page") == best_page:
            ci = r.get("chunk_index", -999)
            if ci not in seen_ci and abs(ci - best_ci) <= 2:
                seen_ci.add(ci)
                parts.append(r["text"].strip())
                for img in r.get("images", []):
                    if img not in all_images:
                        all_images.append(img)

    # Sort parts by chunk_index for reading order
    # Build (ci, text) pairs so we can sort
    ci_parts = [(best_ci, best["text"].strip())]
    for r in results:
        if r.get("page") == best_page:
            ci = r.get("chunk_index", -999)
            if ci not in {best_ci} and abs(ci - best_ci) <= 2:
                ci_parts.append((ci, r["text"].strip()))
    ci_parts.sort(key=lambda x: x[0])
    ordered_parts = [t for _, t in ci_parts]

    # Dedup: remove parts with >80% word overlap to the primary
    primary_words = set(best["text"].lower().split())
    final_parts = []
    seen_content = set()
    for part in ordered_parts:
        key = part[:80].lower()
        if key in seen_content:
            continue
        seen_content.add(key)
        part_words = set(part.lower().split())
        if part == best["text"].strip() or len(primary_words & part_words) / max(len(part_words), 1) < 0.7:
            final_parts.append(part)

    if not final_parts:
        final_parts = [best["text"].strip()]

    # Verbatim assembly: join with double-newline (respects paragraph breaks from PDF)
    answer = "\n\n".join(final_parts)
    # Clean up extra spaces but preserve newlines
    answer = re.sub(r'[ \t]+', ' ', answer)
    # Remove stray leading numbers from page artifacts (e.g. "86 Both..." -> "Both...")
    answer = re.sub(r'^\d{1,3}\s+(?=[A-Z])', '', answer)

    final_pages = sorted(set(p for p in all_pages if p != "?"))
    return answer, final_pages


# ─── Suggestion Utilities ─────────────────────────────────────────────────────

def _extract_key_phrases(text: str, max_phrases=15) -> list:
    """Extract meaningful concepts, filtering out generic single words."""
    raw_words = re.findall(r"\b[A-Za-z]{3,}\b", text) # Ignore short artifacts
    if len(raw_words) < 10: return []

    counts = {}
    word_freq = {}
    for w in raw_words:
        lw = w.lower()
        word_freq[lw] = word_freq.get(lw, 0) + 1

    # 1. Bigrams & Trigrams - Priority 1
    for i in range(len(raw_words) - 1):
        a, b = raw_words[i], raw_words[i+1]
        la, lb = a.lower(), b.lower()
        if la in _STOP or lb in _STOP or la in _GENERIC_PDF_WORDS or lb in _GENERIC_PDF_WORDS:
            continue

        # Concept must be somewhat common in the doc (at least twice)
        bg = f"{la} {lb}"
        counts[bg] = counts.get(bg, 0.0) + (4.0 if a[0].isupper() and b[0].isupper() else 2.5)

    # 2. Add Trigrams
    for i in range(len(raw_words) - 2):
        a, b, c = raw_words[i], raw_words[i+1], raw_words[i+2]
        la, lb, lc = a.lower(), b.lower(), c.lower()
        if la in _STOP or lb in _STOP or lc in _STOP or la in _GENERIC_PDF_WORDS or lb in _GENERIC_PDF_WORDS or lc in _GENERIC_PDF_WORDS:
            continue
        tg = f"{la} {lb} {lc}"
        counts[tg] = counts.get(tg, 0.0) + 5.0

    sorted_phrases = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Filter: Deduplicate (don't show "Intelligence" if "Artificial Intelligence" exists)
    final = []
    seen_text = ""
    for p, _ in sorted_phrases:
        if p not in seen_text:
            final.append(p)
            seen_text += " " + p

    return final[:max_phrases]


def _sent_to_question(sent: str) -> str | None:
    """Convert a declarative sentence to a question ONLY if it's a high-quality definition."""
    sent = sent.strip().rstrip('.')
    if not sent or len(sent) < 20:
        return None
    
    # Pre-filter: Ignore lines that look like page headers or footers (mostly numbers/symbols)
    if len(re.findall(r'[A-Za-z]', sent)) < len(sent) * 0.4:
        return None

    # Strict introductory phrase stripping
    intro_pattern = re.compile(
        r'^(According to [^,]+|In (this|the) [^,]+|For [^,]+|Furthermore|However|Additionally|Moreover|In addition|Notably|Interestingly|Basically|From a [^,]+ perspective|As [^,]+),?\s*',
        re.I
    )
    clean_sent = intro_pattern.sub('', sent).strip()
    
    # Must have a clear "is/are" in the middle, not at the start
    low = clean_sent.lower()
    for kw in [' is ', ' are ', ' refers to ', ' means ', ' consists of ']:
        if kw in low:
            idx = low.find(kw)
            subject = clean_sent[:idx].strip()
            
            # SUBJECT VALIDATION
            # 1. Strip articles
            subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.I).strip()
            # 2. Block single-word subjects (too generic)
            subj_words = subject.split()
            if len(subj_words) < 2:
                continue
            # 3. Block subjects that start with prepositions or pronouns
            if subj_words[0].lower() in {'it', 'they', 'this', 'that', 'from', 'with', 'under', 'between', 'among'}:
                continue
            # 4. Length check
            if len(subject) < 8 or len(subject) > 60:
                continue
                
            template = "What is {}?" if ' is ' in kw else "What are {}?"
            if 'means' in kw or 'refers to' in kw: template = "What does {} mean?"
            
            return template.format(subject.title())

    return None


def _extract_key_phrases(text: str, max_phrases=15) -> list:
    """Extract meaningful concepts, filtering out generic single words."""
    raw_words = re.findall(r"\b[A-Za-z]{3,}\b", text) # Ignore short artifacts
    if len(raw_words) < 10: return []

    counts = {}
    word_freq = {}
    for w in raw_words:
        lw = w.lower()
        word_freq[lw] = word_freq.get(lw, 0) + 1

    # 1. Bigrams & Trigrams - Priority 1
    for i in range(len(raw_words) - 1):
        a, b = raw_words[i], raw_words[i+1]
        la, lb = a.lower(), b.lower()
        if la in _STOP or lb in _STOP or la in _GENERIC_PDF_WORDS or lb in _GENERIC_PDF_WORDS:
            continue
        
        # Concept must be somewhat common in the doc (at least twice)
        bg = f"{la} {lb}"
        counts[bg] = counts.get(bg, 0.0) + (4.0 if a[0].isupper() and b[0].isupper() else 2.5)

    # 2. Add Trigrams
    for i in range(len(raw_words) - 2):
        a, b, c = raw_words[i], raw_words[i+1], raw_words[i+2]
        la, lb, lc = a.lower(), b.lower(), c.lower()
        if la in _STOP or lb in _STOP or lc in _STOP or la in _GENERIC_PDF_WORDS or lb in _GENERIC_PDF_WORDS or lc in _GENERIC_PDF_WORDS:
            continue
        tg = f"{la} {lb} {lc}"
        counts[tg] = counts.get(tg, 0.0) + 5.0

    sorted_phrases = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Filter: Deduplicate (don't show "Intelligence" if "Artificial Intelligence" exists)
    final = []
    seen_text = ""
    for p, _ in sorted_phrases:
        if p not in seen_text:
            final.append(p)
            seen_text += " " + p
            
    return final[:max_phrases]

def generate_suggestions(chunks: list, n: int = 4) -> list:
    """Generate diverse, high-quality questions based on document themes."""
    if not chunks: return []
    
    suggestions = []
    seen = set()
    
    def add_q(q):
        if not q or len(q) < 15: return False
        k = q.lower().strip()
        # Safety block for common bad starts or generic garbage
        blocked = ['what is according', 'what is artificial', 'from a programming', 'is there a', 'the following']
        if any(b in k for b in blocked) or len(k.split()) < 3:
            return False
        if k not in seen:
            seen.add(k)
            suggestions.append(q)
            return True
        return False

    # Extract themes and definitions
    full_text = " ".join(c["text"] for c in chunks[:150])
    concepts = _extract_key_phrases(full_text, max_phrases=20)
    
    # 1. Primary: Use concepts with diverse templates
    templates = [
        "What is {}?", "How does {} work?", "Explain the concept of {}.",
        "What are the main features of {}?", "What is the role of {}?",
        "Tell me more about {}.", "What are the common uses of {}?"
    ]
    
    # Pick a random mix of concepts and templates
    concept_pool = list(concepts)
    random.shuffle(concept_pool)
    
    for concept in concept_pool:
        p_title = concept.title()
        # Try a few templates for each concept until one fits
        for _ in range(3):
            tmpl = random.choice(templates)
            if add_q(tmpl.format(p_title)):
                break
        if len(suggestions) >= n: break

    # 2. Secondary: Fallback to definition-style sentences
    if len(suggestions) < n:
        for chunk in chunks[:50]:
            for sent in sent_tokenize(chunk["text"]):
                q = _sent_to_question(sent)
                if q: add_q(q)
                if len(suggestions) >= n: break
            if len(suggestions) >= n: break

    # 3. Last Resort: Generic but safe document questions
    if len(suggestions) < n:
        backups = ["Can you summarize this document?", "What are the key takeaways?", "Who is the intended audience?"]
        for b in backups:
            add_q(b)
            if len(suggestions) >= n: break

    random.shuffle(suggestions)
    return suggestions[:n]


def generate_followup_suggestions(prev_question: str, prev_answer: str,
                                  chunks: list, index, n: int = 4) -> list:
    """
    Generate contextually relevant follow-up suggestions after a Q&A exchange:
    1. Embed the previous answer to find *nearby-but-different* chunks
    2. Skip chunks that heavily overlap with the already-answered content
    3. Extract fresh questions from those adjacent chunks
    """
    if index is None or not chunks:
        return generate_suggestions(chunks, n)

    # Embed the previous answer to find adjacent topic territory
    combined = f"{prev_question} {prev_answer}"
    q_emb    = _get_model().encode([combined], normalize_embeddings=True, show_progress_bar=False)
    q_emb    = np.array(q_emb).astype('float32')

    k_cands = min(max(n * 6, 24), len(chunks))
    _, indices = index.search(q_emb, k=k_cands)
    cand_idx = [int(i) for i in indices[0] if i != -1]

    # Words already covered by the previous answer
    answered_words = set(w.lower() for w in prev_answer.split() if len(w) > 3)

    suggestions = []
    seen_sents  = set()

    for ci in cand_idx:
        chunk = chunks[ci]
        # Skip chunks that are essentially the same content as the answer
        chunk_words = set(w.lower() for w in chunk["text"].split() if len(w) > 3)
        overlap_ratio = len(answered_words & chunk_words) / max(len(chunk_words), 1)
        if overlap_ratio > 0.55:   # too similar — skip
            continue

        sents = sent_tokenize(chunk["text"])
        for sent in sents:
            key = sent.strip()[:60].lower()
            if key in seen_sents:
                continue
            seen_sents.add(key)

            q = _sent_to_question(sent)
            if q and q not in suggestions and q.lower() != prev_question.lower():
                suggestions.append(q)
            if len(suggestions) >= n:
                break
        if len(suggestions) >= n:
            break

    # Pad with topic-phrase questions if needed
    if len(suggestions) < n:
        full_text = " ".join(c["text"] for c in chunks)
        phrases   = _extract_key_phrases(full_text)
        templates = ["What is {}?", "How does {} work?",
                     "What are the uses of {}?", "Explain {}."]
        for phrase in phrases:
            for tmpl in templates:
                q = tmpl.format(phrase.title())
                if q not in suggestions and prev_question.lower() not in q.lower():
                    suggestions.append(q)
                    break
            if len(suggestions) >= n:
                break

    deduped = list(dict.fromkeys(suggestions))
    return deduped[:n] if deduped else generate_suggestions(chunks, n)

