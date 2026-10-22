
from app.ingestion.embedder import Embedder
from app.retrieval.reranker import rerank_documents
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever

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

try:
    question = "What is the difference between similarity and relevance in RAG?"


    # Retrieve relevant documents from the vector store
    documents = retriever.retrieve(question)

    print(f"Number of relevant chunks: {len(documents)}")

    for result in documents:
        print("\n---")
        print("Source:", result["metadata"].get("file_name"))
        print("Page:", result["metadata"].get("page_number"))
        print("Distance:", round(result["distance"], 4))
        print("Similarity:", round(result["similarity"], 4))
        print("Text:", result["text"][:300])

    

    # print("\n===== CONTEXT FOR LLM =====")
    # context = retriever.build_context(documents)
    # print(context)
    
    # Rerank the retrieved documents based on their relevance to the question
    results = rerank_documents(
        query=question,
        documents=documents,
        top_k=2,
    )

    for index, document in enumerate(results, start=1):
        metadata = document.get("metadata", {})

        print(f"\nRank: {index}")
        print("Source:", metadata.get("file_name", "Unknown"))
        print("Page:", metadata.get("page_number", "Unknown"))

        if "rerank_score" in document:
            print(f"Rerank score: {document['rerank_score']:.4f}")

        print("Distance:", document.get("distance"))
        print("Similarity:", document.get("similarity"))
        print("Text:", document["text"][:300])
    
    
    

finally:
    store.close()


# python -m app.tests.test_retriever