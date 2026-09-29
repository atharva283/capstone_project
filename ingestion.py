'''Bounded, multi-format ingestion with friendly errors for every kind of bad upload.'''
from __future__ import annotations
import codecs
import hashlib
import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator
import pandas as pd
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pypdf prints one warning per slightly damaged object; they are harmless noise here.
logging.getLogger('pypdf').setLevel(logging.ERROR)

SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.csv', '.xlsx'}
SUPPORTED_TEXT = 'Supported types: PDF, TXT, CSV and XLSX.'
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_EXTRACTED_CHARACTERS = 2_000_000
MAX_DOCUMENTS = 20_000
MAX_XLSX_EXPANDED_BYTES = 100 * 1024 * 1024
UPLOAD_DIR = Path(__file__).resolve().parent / 'uploaded_documents'

_ARCHIVE = 'ZIP and other archives cannot be read. Extract the archive first, then upload the PDF, TXT, CSV or XLSX files inside it.'
_WORD = 'Word documents are not supported. In Word choose File > Save As > PDF (or Plain Text) and upload that file instead.'
_IMAGE = 'Images are not supported because text recognition (OCR) is not included. Upload a PDF or TXT file that contains real text.'
_SLIDES = 'PowerPoint files are not supported. Export the slides as a PDF and upload that instead.'
_OLD_EXCEL = 'Old Excel files (.xls) are not supported. Open the file in Excel and use Save As > Excel Workbook (.xlsx) or CSV.'
UNSUPPORTED_HINTS = {
    '.docx': _WORD, '.doc': _WORD, '.odt': _WORD, '.rtf': _WORD,
    '.zip': _ARCHIVE, '.rar': _ARCHIVE, '.7z': _ARCHIVE, '.gz': _ARCHIVE, '.tar': _ARCHIVE,
    '.xls': _OLD_EXCEL, '.xlsm': _OLD_EXCEL, '.pptx': _SLIDES, '.ppt': _SLIDES,
    '.png': _IMAGE, '.jpg': _IMAGE, '.jpeg': _IMAGE, '.gif': _IMAGE, '.bmp': _IMAGE,
    '.tif': _IMAGE, '.tiff': _IMAGE, '.webp': _IMAGE,
    '.exe': 'Programs are not documents and cannot be opened here.',
}
PARSE_FAILURE_HINTS = {
    '.pdf': 'This PDF could not be read. It may be corrupted or not a real PDF.',
    '.xlsx': 'This Excel file could not be read. It may be corrupted, password-protected, or not a real .xlsx file.',
    '.csv': 'This CSV file could not be read. It may be corrupted or binary data renamed to .csv.',
    '.txt': 'This text file could not be read. It may be corrupted or binary data renamed to .txt.',
}


class IngestionError(ValueError):
    '''A file is unsupported, unreadable, or exceeds the ingestion limits.'''


def safe_filename(filename: str) -> str:
    '''Keep only the final path component and harmless characters; never trust client names.'''
    name = Path(filename.replace('\\', '/')).name.strip()
    name = re.sub(r'[^\w.\- ()]', '_', name).lstrip('.')
    return name[:120] or 'document'


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
                raise IngestionError('The expanded Excel workbook exceeds 100 MiB.')
        with pd.ExcelFile(self.path, engine='openpyxl') as workbook:
            for sheet in workbook.sheet_names:
                frame = pd.read_excel(
                    workbook, sheet_name=sheet, dtype=str, keep_default_na=False,
                    nrows=MAX_DOCUMENTS + 1,
                )
                if len(frame) > MAX_DOCUMENTS:
                    raise IngestionError('An Excel sheet exceeds the 20,000-row limit.')
                for position, (_, row) in enumerate(frame.iterrows(), start=2):
                    values = [f'{column}: {value}' for column, value in row.items()
                              if str(value).strip()]
                    if values:
                        yield Document(
                            page_content='\n'.join(values),
                            metadata={'sheet': str(sheet), 'row': position},
                        )


def _text_encoding(data: bytes) -> str | None:
    '''Pick UTF-8, UTF-16 (with BOM) or Windows-1252 (Excel's default CSV); None means binary.'''
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return 'utf-16'
    if b'\x00' in data:
        return None
    for encoding in ('utf-8-sig', 'cp1252'):
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return None


def _parse_failure_message(extension: str, path: Path) -> str:
    if extension == '.pdf':
        try:
            from pypdf import PdfReader
            if PdfReader(str(path)).is_encrypted:
                return 'This PDF is password-protected. Remove the password and upload it again.'
        except Exception:  # noqa: BLE001 - fall through to the generic PDF message
            pass
    return PARSE_FAILURE_HINTS[extension]


def describe_empty(filename: str, size: int) -> str:
    '''Explain why a file that loaded without errors produced no text.'''
    extension = Path(filename).suffix.lower()
    if size == 0:
        return 'is empty (0 bytes)'
    if extension == '.pdf':
        return 'has no selectable text (it may be a scanned image; OCR is not supported)'
    if extension in {'.csv', '.xlsx'}:
        return 'has no data rows'
    return 'contains only blank text'


def load_document(filename: str, data: bytes) -> list[Document]:
    '''Load one upload and replace temporary-path metadata with safe provenance.

    Unsupported types and unreadable files raise IngestionError with a friendly message;
    empty files and files without extractable text return an empty list.
    Callers can continue with other files after either outcome.
    PDF pages are one-based; CSV/Excel rows use spreadsheet numbers (the header is row 1).
    '''
    safe_name = safe_filename(filename)
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        reason = UNSUPPORTED_HINTS.get(
            extension,
            'This file has no extension.' if not extension else f"'{extension}' files are not supported.")
        raise IngestionError(f"'{safe_name}' cannot be used. {reason} {SUPPORTED_TEXT}")
    if not data:
        return []
    if len(data) > MAX_FILE_BYTES:
        raise IngestionError(f"'{safe_name}' is larger than the 20 MiB limit per file.")
    encoding = None
    if extension in {'.txt', '.csv'}:
        encoding = _text_encoding(data)
        if encoding is None:
            raise IngestionError(f"'{safe_name}': {PARSE_FAILURE_HINTS[extension]}")
    file_id = hashlib.sha256(data).hexdigest()
    documents: list[Document] = []
    total_characters = 0
    # TemporaryDirectory closes the file before loaders reopen it on Windows.
    with tempfile.TemporaryDirectory(prefix='document_assistant_') as directory:
        path = Path(directory) / f'upload{extension}'  # never the client-supplied name
        path.write_bytes(data)
        try:
            if extension == '.pdf':
                loader = PyPDFLoader(str(path))
            elif extension == '.txt':
                loader = TextLoader(str(path), encoding=encoding)
            elif extension == '.csv':
                loader = CSVLoader(str(path), encoding=encoding)
            else:
                loader = ExcelLoader(str(path))
            for document in loader.lazy_load():
                text = document.page_content.strip()
                if not text:
                    continue
                total_characters += len(text)
                if total_characters > MAX_EXTRACTED_CHARACTERS or len(documents) >= MAX_DOCUMENTS:
                    raise IngestionError(
                        f"'{safe_name}' exceeds 2 million extracted characters or 20,000 records.")
                # Whitelist scalar metadata instead of persisting PDF internals or temp paths.
                metadata = {'source': safe_name, 'file_id': file_id, 'format': extension[1:]}
                if 'page' in document.metadata:
                    metadata['page'] = int(document.metadata['page']) + 1
                if 'row' in document.metadata:
                    # Spreadsheet-style numbers (header = row 1): CSVLoader counts data rows from 0.
                    metadata['row'] = int(document.metadata['row']) + (2 if extension == '.csv' else 0)
                if 'sheet' in document.metadata:
                    metadata['sheet'] = str(document.metadata['sheet'])
                documents.append(Document(page_content=text, metadata=metadata))
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"'{safe_name}': {_parse_failure_message(extension, path)}") from exc
    return documents


def chunk_documents(
    documents: list[Document], chunk_size: int = 700, chunk_overlap: int = 100,
) -> list[Document]:
    '''Split content while preserving provenance and deterministic chunk IDs.

    Small character chunks reduce MiniLM's token-window truncation risk, but do
    not guarantee it for every language. Tune against your own document corpus.
    '''
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError('Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.')
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        add_start_index=True, separators=['\n\n', '\n', ' ', ''],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        file_id = chunk.metadata.get('file_id', 'unknown')
        identity = f'{file_id}:{index}:{chunk.page_content}'
        chunk.metadata['chunk_id'] = hashlib.sha256(identity.encode('utf-8')).hexdigest()
    return chunks


# ---------------------------------------------------------------- uploaded_documents/ folder
def _session_folder(session_id: str) -> Path:
    if not re.fullmatch(r'docs_[a-f0-9]{32}', session_id):
        raise ValueError('Session id must be docs_ followed by a UUID hex value.')
    return UPLOAD_DIR / session_id


def save_upload_copy(session_id: str, filename: str, data: bytes) -> Path:
    '''Keep a copy of a validated upload in uploaded_documents/<session>/ (safe file name only).'''
    folder = _session_folder(session_id)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_filename(filename)
    target.write_bytes(data)
    return target


def delete_session_uploads(session_id: str) -> None:
    shutil.rmtree(_session_folder(session_id), ignore_errors=True)


def purge_stale_uploads(upload_dir: Path = UPLOAD_DIR) -> None:
    '''Delete copies left by earlier runs. Session ids are random, so nothing can reuse them.'''
    if not upload_dir.is_dir():
        return
    for item in upload_dir.iterdir():
        if item.name == '.gitkeep':
            continue
        shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink(missing_ok=True)
