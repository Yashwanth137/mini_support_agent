import pytest
from rag import RAGPipeline
from config import settings

def test_rag_pipeline_initialization():
    rag = RAGPipeline()
    # Ensure index is created and has vectors
    assert rag.index is not None
    assert len(rag.documents) > 0
    assert rag.index.ntotal == len(rag.documents)

def test_rag_retrieval():
    rag = RAGPipeline()
    docs = rag.retrieve("What is your return policy?")
    assert len(docs) > 0
    # One of the docs should likely contain return policy info,
    # or have a relevant source like 'returns_and_refunds.md'
    sources = [doc.metadata.get("source", "") for doc in docs]
    assert any("return" in src.lower() for src in sources)
