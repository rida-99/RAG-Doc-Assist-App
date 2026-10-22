
from app.retrieval.vector_store import VectorStore

store = VectorStore(
    persist_directory="storage/test_vector_store",
    collection_name="test_documents",
)
chunks = [
    {
        "id": "chunk-test-001",
        "text": "RAG combines document retrieval with language models to generate answers grounded in external knowledge.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 1,
        },
        "embedding": [1.0, 0.0, 0.0],
    },
    {
        "id": "chunk-test-002",
        "text": "Vector databases store numerical representations of text called embeddings. These vectors enable semantic similarity search.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 1,
        },
        "embedding": [0.0, 1.0, 0.0],
    },
    {
        "id": "chunk-test-003",
        "text": "The document ingestion pipeline validates uploaded files before extracting their text and metadata.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 2,
        },
        "embedding": [0.9, 0.1, 0.0],
    },
    {
        "id": "chunk-test-004",
        "text": "Text chunking divides long documents into smaller segments that can be retrieved independently. Overlapping chunks help preserve context.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 2,
        },
        "embedding": [0.8, 0.2, 0.0],
    },
    {
        "id": "chunk-test-005",
        "text": "An embedding model converts text into a numerical vector that represents its semantic meaning.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 3,
        },
        "embedding": [0.1, 0.9, 0.0],
    },
    {
        "id": "chunk-test-006",
        "text": "A retriever searches the vector database for document chunks that are semantically relevant to a user query.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 3,
        },
        "embedding": [0.2, 0.8, 0.0],
    },
    {
        "id": "chunk-test-007",
        "text": "A reranker reorders retrieved chunks according to their relevance to the original question, improving the quality of the context.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 4,
        },
        "embedding": [0.7, 0.2, 0.1],
    },
    {
        "id": "chunk-test-008",
        "text": "The prompt builder combines the user's question with the retrieved document context before sending it to the language model.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 4,
        },
        "embedding": [0.6, 0.3, 0.1],
    },
    {
        "id": "chunk-test-009",
        "text": "The language model generates an answer using the provided context. A well-designed RAG system should avoid unsupported claims.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 5,
        },
        "embedding": [0.5, 0.4, 0.1],
    },
    {
        "id": "chunk-test-010",
        "text": "Document metadata can include the document identifier, original filename, file type, page number, and upload timestamp.",
        "metadata": {
            "doc_id": "doc-test-001",
            "file_name": "example.txt",
            "file_type": ".txt",
            "page_number": 5,
        },
        "embedding": [0.1, 0.2, 0.7],
    },
    {
        "id": "chunk-test-011",
        "text": "File validation checks the file extension, size, and content before saving an uploaded document for processing.",
        "metadata": {
            "doc_id": "doc-test-002",
            "file_name": "security.txt",
            "file_type": ".txt",
            "page_number": 1,
        },
        "embedding": [0.0, 0.2, 0.8],
    },
    {
        "id": "chunk-test-012",
        "text": "A document identifier allows the application to associate multiple chunks with the same source document and support document-level deletion.",
        "metadata": {
            "doc_id": "doc-test-002",
            "file_name": "security.txt",
            "file_type": ".txt",
            "page_number": 2,
        },
        "embedding": [0.1, 0.1, 0.8],
    },
]

try:
    count = store.upsert_chunks(chunks)
    print("Inserted:", count)
    print("Total chunks:", store.count_chunks())
    print("Documents:", store.list_documents())

    results = store.search_by_embedding(
        query_embedding=[0.9, 0.1, 0.0],
        top_k=2,
    )

    for result in results:
        print(result["text"], result["distance"])

    deleted = store.delete_document("doc-test-001")
    deleted = store.delete_document("doc-test-002")
    print("Deleted:", deleted)

finally:
    store.close()
