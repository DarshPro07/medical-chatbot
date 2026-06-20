# create_memory_for_llm.py
# pip install langchain-community langchain-huggingface faiss-cpu python-dotenv
# Optional (better text extraction / OCR-capable fallback):
# pip install pymupdf

from pathlib import Path
import os
import json
import hashlib

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_community.document_loaders import PyMuPDFLoader  # fallback if available
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------- Config ----------------- #
DATA_PATH = Path("data")
DB_FAISS_PATH = "vectorstore/db_faiss"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_BYTES = 1024                   # <1KB: likely empty/corrupt
STRICT_PDF_MAGIC = True            # require PDF magic in first bytes
MAGIC_SEARCH_BYTES = 2048          # search window for "%PDF-"
DEDUP_CHUNKS = True                # remove duplicate chunks across files
# Set a stronger embedder if you want better retrieval quality (e.g., BGE)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_ENCODE_KW = {"normalize_embeddings": True, "batch_size": 64}
# ----------------------------------------- #

load_dotenv()

def list_pdfs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.pdf"))

def is_valid_pdf(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < MIN_BYTES:
            return False
        with open(path, "rb") as f:
            head = f.read(MAGIC_SEARCH_BYTES)
            if STRICT_PDF_MAGIC and (b"%PDF-" not in head):
                return False
        return True
    except Exception:
        return False

def load_pages_with_fallback(pdf_path: Path):
    # Try PyPDFLoader first (fast, good metadata)
    try:
        pages = PyPDFLoader(str(pdf_path)).load()
        if pages:
            return pages, "PyPDFLoader"
    except Exception as e:
        print(f"[WARN] {pdf_path.name}: PyPDFLoader failed: {e}")

    # Fallback to PyMuPDFLoader (better layout/text in many PDFs)
    if PYMUPDF_AVAILABLE:
        try:
            pages = PyMuPDFLoader(str(pdf_path)).load()
            if pages:
                return pages, "PyMuPDFLoader"
        except Exception as e:
            print(f"[WARN] {pdf_path.name}: PyMuPDFLoader failed: {e}")

    # No pages
    return [], None

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    # Ensure metadata keeps file_name and page (PyPDFLoader already sets page/source)
    for c in chunks:
        src = c.metadata.get("source", "")
        c.metadata["file_name"] = Path(src).name if src else c.metadata.get("file_name", "")
    # Drop empties
    return [c for c in chunks if c.page_content and c.page_content.strip()]

def hash_text(s: str) -> str:
    return hashlib.sha1(" ".join(s.split()).encode("utf-8")).hexdigest()

def dedup_chunks(chunks):
    seen = set()
    unique = []
    for c in chunks:
        h = hash_text(c.page_content)
        if h in seen:
            continue
        seen.add(h)
        unique.append(c)
    return unique

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs=EMBED_ENCODE_KW)

def load_existing_index(db_path: str, embedder):
    # If you want to append to an existing index safely and model matches
    base_dir = os.path.dirname(db_path)
    meta_file = os.path.join(base_dir, "meta.json")
    if os.path.isdir(base_dir) and os.path.exists(os.path.join(base_dir, "index.faiss")):
        # Optional check: ensure same embedding model
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("embedding_model") != EMBED_MODEL:
                print("[WARN] Existing index was built with a different embedding model. Rebuilding is recommended.")
        except Exception:
            pass
        try:
            return FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"[WARN] Could not load existing index: {e}")
    return None

def save_index(db, db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    db.save_local(db_path)
    # Save simple metadata to help future checks
    meta = {"embedding_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP}
    with open(os.path.join(os.path.dirname(db_path), "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] FAISS saved to {db_path}")

def main():
    if not DATA_PATH.exists():
        print(f"[ERROR] Data folder not found: {DATA_PATH.resolve()}")
        return

    pdf_files = list_pdfs(DATA_PATH)
    if not pdf_files:
        print(f"[WARN] No PDFs found in {DATA_PATH.resolve()}")
        return

    embedder = get_embedding_model()

    # Load existing index if present
    db = load_existing_index(DB_FAISS_PATH, embedder)

    total_files = 0
    total_pages = 0
    total_chunks = 0
    skipped_files = 0

    for pdf in pdf_files:
        print(f"\n[PROCESS] {pdf.name}")

        if not is_valid_pdf(pdf):
            print(f"[SKIP] {pdf.name}: invalid/empty/non-PDF")
            skipped_files += 1
            continue

        pages, loader_used = load_pages_with_fallback(pdf)
        if not pages:
            print(f"[SKIP] {pdf.name}: no pages extracted")
            skipped_files += 1
            continue

        chunks = chunk_pages(pages)
        if not chunks:
            print(f"[SKIP] {pdf.name}: no extractable text (possibly image-only; consider OCR)")
            skipped_files += 1
            continue

        if DEDUP_CHUNKS:
            before = len(chunks)
            chunks = dedup_chunks(chunks)
            after = len(chunks)
            if after < before:
                print(f"[INFO] Deduped chunks: {before} -> {after}")

        try:
            if db is None:
                print(f"[EMBED] ({loader_used}) Creating FAISS with {len(chunks)} chunks")
                db = FAISS.from_documents(chunks, embedder)
            else:
                print(f"[EMBED] ({loader_used}) Adding {len(chunks)} chunks")
                db.add_documents(chunks)

            total_files += 1
            total_pages += len(pages)
            total_chunks += len(chunks)
            print(f"[OK] {pdf.name}: pages={len(pages)}, chunks={len(chunks)}")
        except Exception as e:
            print(f"[ERROR] Embedding failed for {pdf.name}: {e}")
            # Continue to next file
            continue

    if db is None:
        print(f"\n[RESULT] No valid documents embedded. Skipped files: {skipped_files}")
        return

    save_index(db, DB_FAISS_PATH)
    print(f"[STATS] files={total_files}, pages={total_pages}, chunks={total_chunks}, skipped={skipped_files}")

if __name__ == "__main__":
    main()