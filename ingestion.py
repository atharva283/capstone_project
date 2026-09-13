'''Bounded, multi-format ingestion; uploaded bytes never become permanent files.'''
from __future__ import annotations
import hashlib
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator
import pandas as pd
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx"}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 2_000_000
MAX_DOCUMENTS = 20_000
MAX_XLSX_EXPANDED_BYTES = 100 * 1024 * 1024
class IngestionError(ValueError):
    '''A file is unsupported, unreadable, or exceeds the ingestion limits.'''

class ExcelLoader(BaseLoader):
    '''LangChain loader emitting one document per nonempty Excel row.

    A custom BaseLoader avoids the heavyweight optional Unstructured dependency.
    Headers, sheet names, and original row numbers remain available for citations.
    Only .xlsx is supported; legacy .xls requires a separate parser.
    '''

    def __init__(self, path: str) -> None:
        self.path = path
    def lazy_load(self) -> Iterator[Document]:
        # Check expanded size before openpyxl parses the ZIP-based workbook.
        with zipfile.ZipFile(self.path) as archive:
            if sum(item.file_size for item in archive.infolist()) > MAX_XLSX_EXPANDED_BYTES:
                raise IngestionError("The expanded Excel workbook exceeds 100 MiB.")
        with pd.ExcelFile(self.path, engine="openpyxl") as workbook:
            for sheet in workbook.sheet_names:
                frame = pd.read_excel(
                    workbook, sheet_name=sheet, dtype=str, keep_default_na=False,
                    nrows=MAX_DOCUMENTS + 1,
                )
                if len(frame) > MAX_DOCUMENTS:
                    raise IngestionError("An Excel sheet exceeds the 20,000-row limit.")
                for position, (_, row) in enumerate(frame.iterrows(), start=2):
                    values = [f"{column}: {value}" for column, value in row.items()
                              if str(value).strip()]
                    if values:
                        yield Document(
                            page_content="\n".join(values),
                            metadata={"sheet": str(sheet), "row": position},
                        )


def load_document(filename: str, data: bytes) -> list[Document]:
    '''Load one upload and replace temporary-path metadata with safe provenance.

    Empty files and files containing only whitespace return an empty list.
    Parser failures raise IngestionError; callers can continue with other files.
    PDF pages and CSV/Excel rows use one-based numbers in returned metadata.
    '''
    if not data:
        return []
    if len(data) > MAX_FILE_BYTES:
        raise IngestionError("Each upload must be at most 20 MiB.")
    # Never use the client-supplied filename as a disk path.
    safe_name = Path(filename.replace("\\", "/")).name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise IngestionError("Supported formats: PDF, UTF-8 TXT/CSV, and XLSX.")
    file_id = hashlib.sha256(data).hexdigest()
    documents: list[Document] = []
    total_characters = 0
    try:
        # TemporaryDirectory closes the file before loaders reopen it on Windows.
        with tempfile.TemporaryDirectory(prefix="document_assistant_") as directory:
            path = Path(directory) / f"upload{extension}"
            path.write_bytes(data)
            if extension == ".pdf":
                loader = PyPDFLoader(str(path))
            elif extension == ".txt":
                loader = TextLoader(str(path), encoding="utf-8-sig")
            elif extension == ".csv":
                loader = CSVLoader(str(path), encoding="utf-8-sig")
            else:
                loader = ExcelLoader(str(path))
            for document in loader.lazy_load():
                text = document.page_content.strip()
                if not text:
                    continue
                total_characters += len(text)
                if total_characters > MAX_EXTRACTED_CHARACTERS or len(documents) >= MAX_DOCUMENTS:
                    raise IngestionError("File exceeds 2 million extracted characters or 20,000 records.")
                # Whitelist scalar metadata instead of persisting PDF internals or temp paths.
                metadata = {"source": safe_name, "file_id": file_id, "format": extension[1:]}
                if "page" in document.metadata:
                    metadata["page"] = int(document.metadata["page"]) + 1
                if "row" in document.metadata:
                    metadata["row"] = int(document.metadata["row"]) + (extension == ".csv")
                if "sheet" in document.metadata:
                    metadata["sheet"] = str(document.metadata["sheet"])
                documents.append(Document(page_content=text, metadata=metadata))
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            "Cannot parse this file. Check its format, UTF-8 encoding, or PDF password protection."
        ) from exc
    return documents

def chunk_documents(
    documents: list[Document], chunk_size: int = 700, chunk_overlap: int = 100,
) -> list[Document]:
    '''Split content while preserving provenance and deterministic chunk IDs.

    Small character chunks reduce MiniLM's token-window truncation risk, but do
    not guarantee it for every language. Tune against your own document corpus.
    '''
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        add_start_index=True, separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        file_id = chunk.metadata.get("file_id", "unknown")
        identity = f"{file_id}:{index}:{chunk.page_content}"
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return chunks
