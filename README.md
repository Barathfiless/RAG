# AI PDF Assistant :book::robot:

A completely local, highly accurate Retrieval-Augmented Generation (RAG) assistant for querying PDF documents. This application processes your PDFs entirely offline, preserving the formatting, layout, extracting images, and building an interactive learning environment directly on your machine.

---

## 🌟 Key Features

* **"God-Mode" Accuracy:** Stitches context chunks together to return verbatim, paragraph-perfect extractions exactly as they appear in the source PDF.
* **Smart Image Extraction:** Pulls images, figures, and charts from the PDF natively.
* **Intelligent Visual Intent:** Only displays images alongside text when your prompt is visually relevant (e.g., "Show me the chart").
* **Local OCR Engine:** Drag and drop an image into the chat; the system reads the text using local AI (EasyOCR + PyTorch) and answers questions based on it.
* **Advanced Learning Tools:**
  * **Quiz Mode:** Auto-generates multiple-choice quizzes (with realistic distractors) from definition-heavy sections.
  * **Smart Document Outline:** Reverse-engineers headings and builds a clickable Table of Contents.
  * **Auto-Glossary:** Scans the text for domain-specific jargon and generates an alphabetical dictionary.
* **100% Offline & Private:** No API keys, no OpenAI, no cloud data transmission. Everything runs locally on your hardware.

---

## 🛠️ Tech Stack & Dependencies

### What is installed, and why?

1. **`Flask`**
   * **Why:** The lightweight web framework used to serve the backend API and render the HTML frontend.
   * **Use:** Handles routing (e.g., `/upload`, `/ask`, `/quiz`), accepts POST requests from the browser, and returns JSON responses or rendered UI.

2. **`PyMuPDF` (`fitz`)**
   * **Why:** The fastest, most robust PDF parsing library available for Python.
   * **Use:** Used to tear the PDF apart. Extracts raw text, detects paragraph boundaries, and isolates and extracts embedded images pixel-by-pixel.

3. **`sentence-transformers`**
   * **Why:** Provides state-of-the-art Natural Language Processing (NLP) models locally.
   * **Use:** We import two models:
     * *Bi-Encoder (`all-MiniLM-L6-v2`)*: Converts text chunks into dense numeric vectors (embeddings).
     * *Cross-Encoder (`ms-marco-TinyBERT-L-2-v2`)*: Re-ranks search results with extremely high precision to ensure the "God-Mode" accuracy levels.

4. **`faiss-cpu` (Facebook AI Similarity Search)**
   * **Why:** High-performance vector database.
   * **Use:** Stores the text embeddings created by `sentence-transformers`. When you ask a question, FAISS performs lightning-fast geometric similarity searches across millions of vectors in milliseconds.

5. **`easyocr` & `torch`**
   * **Why:** Local Optical Character Recognition.
   * **Use:** Replaces error-prone external engines like Tesseract. By leveraging PyTorch, if you upload an image to the chat, EasyOCR natively reads the pixels and turns them into text strings the search engine can understand.

6. **`rank_bm25`**
   * **Why:** Traditional keyword-matching algorithm.
   * **Use:** Because vector AI sometimes misses exact keyword names, BM25 creates a "hybrid" search mechanism. It ensures that if you query a highly specific serial number or jargon phrase, it won't be missed.

7. **`nltk`**
   * **Why:** Natural Language Toolkit.
   * **Use:** Used for sophisticated chunking. Instead of slicing text arbitrarily by character count (which breaks words), NLTK understands grammar and splits text accurately by *sentences*, meaning the context window is never fragmented mid-thought.

---

## 🧠 How It Works Under The Hood

### 1. Document Ingestion & Image Processing
When you upload a PDF:
* `utils.py` opens the file via `PyMuPDF`.
* It iterates through the document, saving embedded `.png` or `.jpeg` files to a secure local folder.
* It uses `NLTK` to chunk the text into smaller blocks (e.g., 5 sentences per chunk with a 2-sentence overlap) to ensure continuity.
* The chunks are vectorized using `sentence-transformers` and indexed into the `FAISS` database.

### 2. Drafting Responses from the PDF (The "Search")
When you type a message:
1. **Geometric Search:** The exact keywords are converted to vectors, and FAISS fetches the top 40 candidates that geometrically match the meaning of your question.
2. **Hybrid & Exact Boost:** BM25 runs alongside it to score exact-word matches. A tiered boosting system adds points for Section Headers and Verbatim matches.
3. **Cross-Encoder Re-Ranking:** The top 15 results are fed into a Heavy NLP model that grades them contextually from 0 to 1 based on how likely they are to actually answer the specific query.
4. **Verbatim Stitching:** In `synthesize_answer`, the system locates the winning result, finds its "neighbors" on the exact same PDF page, and literally stitches the paragraphs back together. This ensures responses aren't robotic summaries—they are identical, layout-preserved quotes from the text.

### 3. Speech Recognition (Frontend)
When you click the microphone icon:
* The web browser triggers the HTML5 `Web Speech API` natively built into modern browsers (Chrome/Edge/Safari).
* As you speak, the browser transcribes the audio into text strings.
* This feature runs completely client-side in Javascript (`index.html`) without sending audio files to the Python server, ensuring instant transcription and zero data usage overhead.

---

## 🚀 Getting Started

### Prerequisites
You will need Python 3.10+ installed.

### 1. Set up the Environment
Open your terminal and create a virtual environment:
```bash
python -m venv venv
```
Activate it:
* **Windows**: `venv\Scripts\activate`
* **Mac/Linux**: `source venv/bin/activate`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If you are setting this up manually without `requirements.txt`, install the packages listed in the **Tech Stack** section above).*

### 3. Run the Backend Server
Start the Flask application:
```bash
python app.py
```

### 4. Open the Interface
Once running, open your web browser and navigate to the local host address printed in the terminal:
```
http://127.0.0.1:5000
```

Upload a PDF, let the indexer process the database, and begin your interactive God-Mode RAG experience!
