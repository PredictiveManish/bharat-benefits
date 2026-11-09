"""
Knowledge base service using ChromaDB for semantic retrieval.
Replaces Moss with an open-source, self-hosted vector store.
"""

import os
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# -- Config -------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = os.getenv(
    "CHROMA_DB_PATH",
    str(SCRIPT_DIR / "chroma_data"),
)
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "bharat-benefits")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


class KnowledgeBase:
    """ChromaDB-based knowledge base for Indian government scheme documents."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        self.collection_name = collection_name or CHROMA_COLLECTION
        self.top_k = top_k or TOP_K

        # Load embedding model
        print(f"   Loading embedding model: {embedding_model or EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(embedding_model or EMBEDDING_MODEL)

        # Initialize ChromaDB client (persistent, local)
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def chunk_document(self, text: str, source: str, scheme_name: str, category: str) -> list[dict]:
        """Split a document into overlapping chunks for better retrieval."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        chunk_ids = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Further split long paragraphs
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) > CHUNK_SIZE and current_chunk:
                    chunk_ids.append(f"{source}_{len(chunks)}")
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source,
                        "scheme_name": scheme_name,
                        "category": category,
                    })
                    current_chunk = sentence
                else:
                    current_chunk = f"{current_chunk} {sentence}".strip()

            if current_chunk:
                chunk_ids.append(f"{source}_{len(chunks)}")
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": source,
                    "scheme_name": scheme_name,
                    "category": category,
                })

        return chunks

    def ingest_documents(self, documents: list[dict]) -> int:
        """
        Ingest documents into the knowledge base.

        Each document dict should have:
          - id: unique identifier
          - text: document content
          - source: source filename
          - scheme_name: name of the scheme
          - category: category (health/farmers/education/etc.)
          - state: state name or "central"
          - helpline: helpline number (optional)
          - portal_url: official portal URL (optional)

        Returns the number of chunks ingested.
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for doc in documents:
            chunks = self.chunk_document(
                text=doc["text"],
                source=doc["source"],
                scheme_name=doc.get("scheme_name", ""),
                category=doc.get("category", "general"),
            )
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk["text"])
                all_metadatas.append({
                    "source": chunk["source"],
                    "scheme_name": chunk["scheme_name"],
                    "category": chunk["category"],
                    "state": doc.get("state", "central"),
                    "helpline": doc.get("helpline", ""),
                    "portal_url": doc.get("portal_url", ""),
                })
                all_ids.append(f"{doc['id']}_{i}")

        if not all_chunks:
            print("   No chunks to ingest.")
            return 0

        # Generate embeddings
        print("   Generating embeddings ...")
        embeddings = self.embedder.encode(all_chunks).tolist()

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            metadatas=all_metadatas,
            embeddings=embeddings,
        )

        print(f"   Ingested {len(all_chunks)} chunk(s) from {len(documents)} document(s).")
        return len(all_chunks)

    def query(self, question: str) -> list[dict]:
        """
        Query the knowledge base for relevant chunks.

        Returns a list of dicts with keys: text, source, scheme_name, category, state, helpline, portal_url, score.
        """
        print("   searching ...")

        # Generate embedding for the question
        question_embedding = self.embedder.encode([question])[0].tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        parts = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            parts.append({
                "text": doc,
                "source": metadata.get("source", "unknown"),
                "scheme_name": metadata.get("scheme_name", ""),
                "category": metadata.get("category", ""),
                "state": metadata.get("state", "central"),
                "helpline": metadata.get("helpline", ""),
                "portal_url": metadata.get("portal_url", ""),
                "score": round(1 - distance, 4),
            })

        return parts

    def get_context_string(self, results: list[dict]) -> str:
        """Format query results into a context string for the LLM."""
        if not results:
            return "No relevant information found in the knowledge base."

        parts = []
        for r in results:
            source = r.get("source", r["source"])
            scheme = r.get("scheme_name", "")
            parts.append(
                f"[Source: {source} | Scheme: {scheme}]\n{r['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def get_or_create_collection(self) -> dict:
        """Get collection info for status reporting."""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "document_count": count,
            "path": CHROMA_DB_PATH,
        }

    def clear_collection(self) -> None:
        """Delete and recreate the collection (for re-indexing)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"   Collection '{self.collection_name}' cleared.")
