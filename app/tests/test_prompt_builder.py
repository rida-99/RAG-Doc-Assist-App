
from app.ingestion.embedder import Embedder
from app.retrieval.retriever import Retriever

from app.retrieval.reranker import rerank_documents
from app.generation.prompt_builder import build_prompt
from app.retrieval.vector_store import VectorStore

question = (
    "What is the difference between similarity and relevance?"
)

# 1. Retrieve candidate chunks.

embedder = Embedder()

store = VectorStore(
    persist_directory="storage/test_vector_store",
    collection_name="test_documents",
)


retriever = Retriever(
    vector_store=store,
    embedder=embedder,
    top_k=5,
    fetch_k=15,
    max_distance=0.35,
)
documents = retriever.retrieve(question)

# print(f"Documents: {documents}")

# 2. Rerank the retrieved chunks.
ranked_documents = rerank_documents(
    query=question,
    documents=documents,
    top_k=3,
)
# print(f"Ranked Documents: {ranked_documents}")

# 3. Build the final prompt.
prompt = build_prompt(
    query=question,
    documents=ranked_documents,
    max_context_chars=12000,
)

print(prompt)

# 4. Send the prompt to your configured LLM provider.
# The actual call depends on your llm_router.py implementation.
