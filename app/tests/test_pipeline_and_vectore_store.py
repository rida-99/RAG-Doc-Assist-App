
import asyncio
import os
from pathlib import Path
import sys
from pprint import pprint
from fastapi import UploadFile

from app.ingestion.pipeline import DocumentIngestionPipeline
from app.retrieval.vector_store import VectorStore


if len(sys.argv) < 2:
    print("Usage: python pipeline.py <file_path>")
    sys.exit(1)

env_path = os.getenv("TEST_FILE", "./app/ingestion/RAG text.pdf")
file_path = Path(env_path)


upload_file = UploadFile(filename=file_path.name, file=open(file_path, "rb"))

pipeline = DocumentIngestionPipeline()
result = asyncio.run(pipeline.ingest(upload_file))
# pprint(result)




store = VectorStore(
    persist_directory="storage/test_vector_store",
    collection_name="test_documents",
)
count = store.upsert_chunks(result["chunks"])
print(f"Successfully stored {count} chunks.")

print("Number of chunks:", len(result["chunks"]))
print("First chunk keys:", result["chunks"][0].keys())
print("First chunk metadata:", result["chunks"][0]["metadata"])

# run with
# $env:TEST_FILE=".\app\ingestion\RAG text.pdf"; python -m unittest app.tests.test_pipeline_and_vectore_store