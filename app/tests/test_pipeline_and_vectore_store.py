
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import sys

from fastapi import UploadFile

from app.ingestion.pipeline import DocumentIngestionPipeline
from app.retrieval.vector_store import VectorStore


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m app.tests.test_pipeline_and_vectore_store <file_path>"
        )
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    store = None

    try:
        # Step 2: Run the ingestion pipeline.
        pipeline = DocumentIngestionPipeline()

        with file_path.open("rb") as input_file:
            upload_file = UploadFile(
                filename=file_path.name,
                file=input_file,
            )

            result = asyncio.run(
                pipeline.ingest(upload_file)
            )

        chunks = result["chunks"]

        print("Ingestion completed successfully.")
        print("Input file:", file_path.name)
        print("Number of chunks:", len(chunks))

        if chunks:
            print("First chunk keys:", chunks[0].keys())
            print(
                "First chunk metadata:",
                chunks[0].get("metadata", {}),
            )

        # Step 3: Store chunks in the vector store.
        store = VectorStore(
            persist_directory="storage/test_vector_store",
            collection_name="test_documents",
        )

        count = store.upsert_chunks(chunks)

        print(f"Successfully stored {count} chunks.")

        # Step 4: Prepare data for JSON serialization.
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(file_path),
            "file_name": file_path.name,
            "ingestion": {
                "number_of_chunks": len(chunks),
                "result": result,
                "chunks": chunks,
            },
            "vector_store": {
                "persist_directory": "storage/test_vector_store",
                "collection_name": "test_documents",
                "number_of_chunks_stored": count,
            },
        }

        # Step 5: Create the output directory.
        output_dir = Path("app/tests/output/pipeline_and_vector_store_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 6: Save results to a JSON file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            output_dir / f"ingestion_{timestamp}.json"
        )

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(
                output_data,
                file,
                ensure_ascii=False,
                indent=4,
                default=str,
            )

        print(f"\nResults saved to: {output_file}")

    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()

# run with
# $env:TEST_FILE=".\app\ingestion\RAG text.pdf"; python -m unittest app.tests.test_pipeline_and_vectore_store