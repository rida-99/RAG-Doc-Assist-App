# app/ingestion/validator.py
import hashlib
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.core.config import settings

MAGIC_BYTES = {
    b"%PDF": ".pdf",
    b"PK\x03\x04": ".docx",   # also .pptx — both are ZIP-based
}

async def validate_and_save(file: UploadFile) -> tuple[Path, str]:
    """Validate upload, persist to disk, return (path, sha256)."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.allowed_file_extensions:
        raise HTTPException(400, f"File type {suffix!r} not allowed.")

    content = await file.read() 

    # Size check
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(413, "File too large.")

    # Magic-byte check (don't trust the extension alone)
    for magic, expected_ext in MAGIC_BYTES.items():
        if content.startswith(magic) and suffix not in (expected_ext, ".pptx"):
            raise HTTPException(400, "File extension doesn't match content.")

    # Deduplication via content hash
    sha256 = hashlib.sha256(content).hexdigest() # compute hash of file content
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.processed_dir / f"{sha256}{suffix}"  # join path and filename
    if not dest.exists():
        dest.write_bytes(content) # binary file write
    else:
        print(f"File already exists: {dest}")
        print(f"SHA256 already exists: {sha256}")
        raise HTTPException(409, "File already exists.")
        
    return dest, sha256

# test validator
if __name__ == "__main__":
    import asyncio
    import sys
    from pprint import pprint
    from fastapi import UploadFile

    if len(sys.argv) < 2:
        print("Usage: python validator.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    upload_file = UploadFile(filename=file_path.name, file=open(file_path, "rb"))

    dest, sha256 = asyncio.run(validate_and_save(upload_file))
    pprint({"dest": str(dest), "sha256": sha256})

# run with
# python -m app.ingestion.validator ".\app\ingestion\RAG text.pdf"