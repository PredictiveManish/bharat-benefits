"""
ingest.py - Bharat Benefits Knowledge Base Ingestion
=====================================================
Reads scheme documents from data/schemes/, chunks them, and populates
the ChromaDB vector store.

Usage:
    python ingest.py                  # ingest all schemes
    python ingest.py --clear          # clear existing index first
    python ingest.py --category health # ingest only health schemes

Environment variables (via .env):
    SARVAM_API_KEY        - Sarvam API key (for STT/TTS, not ingestion)
    CHROMA_DB_PATH        - ChromaDB storage path (default: ./chroma_data)
    CHROMA_COLLECTION     - Collection name (default: bharat-benefits)
    EMBEDDING_MODEL       - Sentence transformer model (default: all-MiniLM-L6-v2)
    TOP_K                 - Number of results per query (default: 3)
    CHUNK_SIZE            - Max chars per chunk (default: 500)
    CHUNK_OVERLAP         - Overlap between chunks (default: 50)
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment
load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMES_DIR = SCRIPT_DIR / "data" / "schemes"


def _load_markdown_files(category: str = None) -> list[dict]:
    """
    Walk SCHEMES_DIR and return document dicts.

    Supports subdirectories by category (health, farmers, education, etc.).
    Each .md file should contain structured scheme information.
    """
    if not SCHEMES_DIR.exists():
        print(f"\n[ERROR] Schemes directory not found: {SCHEMES_DIR}", file=sys.stderr)
        sys.exit(1)

    md_files = []
    if category:
        cat_dir = SCHEMES_DIR / category
        if not cat_dir.exists():
            print(f"\n[ERROR] Category directory not found: {cat_dir}", file=sys.stderr)
            sys.exit(1)
        md_files = sorted(cat_dir.glob("*.md"))
    else:
        # Recursively find all .md files
        for root, dirs, files in os.walk(SCHEMES_DIR):
            for f in files:
                if f.endswith(".md"):
                    md_files.append(Path(root) / f)
        md_files = sorted(md_files)

    if not md_files:
        print(f"\n[ERROR] No markdown files found in {SCHEMES_DIR}", file=sys.stderr)
        sys.exit(1)

    docs = []
    for path in md_files:
        text = path.read_text(encoding="utf-8").strip()
        # Derive category from parent directory
        parent = path.parent.name
        # Derive scheme name from filename
        scheme_name = path.stem.replace("-", " ").title()

        docs.append({
            "id": path.stem,
            "text": text,
            "source": path.name,
            "scheme_name": scheme_name,
            "category": parent,
            "state": "central" if parent == "central" else parent,
        })

    return docs


def main():
    parser = argparse.ArgumentParser(description="Ingest scheme documents into ChromaDB")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before ingesting",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Only ingest documents from a specific category (e.g., health, farmers)",
    )
    args = parser.parse_args()

    # Import here to avoid loading embedding model unnecessarily
    from bharat_benefits.services.knowledge_base import KnowledgeBase

    # Load documents
    print(f"\nLoading scheme documents from: {SCHEMES_DIR}")
    raw_docs = _load_markdown_files(category=args.category)
    print(f"   Found {len(raw_docs)} document(s):")
    for d in raw_docs:
        print(f"   - {d['source']} ({d['scheme_name']})")

    # Initialize knowledge base
    kb = KnowledgeBase()

    # Clear if requested
    if args.clear:
        kb.clear_collection()

    # Ingest documents
    chunk_count = kb.ingest_documents(raw_docs)

    # Report status
    info = kb.get_or_create_collection()
    print(f"\nKnowledge base ready:")
    print(f"   Collection : {info['name']}")
    print(f"   Documents  : {info['document_count']} chunk(s)")
    print(f"   Path       : {info['path']}")

    print("\nNext step: run the bot with")
    print('  python main.py --text "I am a small farmer. Which scheme can help me?"\n')


if __name__ == "__main__":
    main()
