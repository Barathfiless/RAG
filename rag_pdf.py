import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

# Load PDF
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Split into chunks
def chunk_text(text):

    sentences = sent_tokenize(text)

    chunk_size = 4
    chunks = []

    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create FAISS index
def create_vector_store(chunks):

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


# Search
def search(question, index, chunks):

    query_embedding = model.encode([question])

    distances, indices = index.search(np.array(query_embedding), k=3)

    results = [chunks[i] for i in indices[0]]

    return results


# Main pipeline
pdf_text = load_pdf("ai_notes.pdf")

chunks = chunk_text(pdf_text)

index = create_vector_store(chunks)

print("PDF indexed successfully!")

while True:

    question = input("\nAsk a question: ")

    results = search(question, index, chunks)

    print("\nMost relevant sections:\n")

    for r in results:
        print(r)
        print()