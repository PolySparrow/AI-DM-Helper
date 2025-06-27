import os
import csv
import json
import re
import unicodedata
import string
import numpy as np
from typing import Optional

import PyPDF2
from docx import Document
import openpyxl
import openai
import config

# ========== PREPROCESSING ==========
def preprocess_text(text, remove_patterns=None):
    text = text.lower()
    text = ''.join(filter(lambda x: x in string.printable, text))
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    if remove_patterns:
        for pat in remove_patterns:
            text = re.sub(pat, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('–', '-')
    return text

# ========== FILE EXTRACTION ==========
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_page_text(pdf_path: str, page_number: int) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        if 0 <= page_number < len(reader.pages):
            return reader.pages[page_number].extract_text() or ""
        else:
            return ""

def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# ========== DOCX SECTION CHUNKING ==========
def extract_sections_from_docx(docx_path: str):
    """
    Returns a list of (heading_dict, section_text) for each section in the DOCX,
    using Heading styles.
    """
    doc = Document(docx_path)
    sections = []
    current_heading = None
    current_section = []
    for para in doc.paragraphs:
        style = para.style.name if para.style else ""
        text = para.text.strip()
        if not text:
            continue
        if style.startswith("Heading"):
            if current_section and current_heading:
                sections.append(({"Section": current_heading}, "\n".join(current_section)))
            current_heading = text
            current_section = []
        else:
            current_section.append(text)
    if current_section and current_heading:
        sections.append(({"Section": current_heading}, "\n".join(current_section)))
    return sections

# ========== TABLE (CSV/XLSX) EXTRACTION ==========
def extract_chunks_from_table(
    file_path: str,
    heading_columns: Optional[list] = None,
    filter_dict: Optional[dict] = None
):
    ext = os.path.splitext(file_path)[1].lower()
    chunks = []

    if ext == ".csv":
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if filter_dict:
                    skip = False
                    for k, v in filter_dict.items():
                        if row.get(k) != v:
                            skip = True
                            break
                    if skip:
                        continue
                headings = {col: row.get(col) for col in heading_columns} if heading_columns else None
                text = ', '.join(f"{k}: {v}" for k, v in row.items())
                chunks.append((headings, text))
    elif ext == ".xlsx":
        wb = openpyxl.load_workbook(file_path, data_only=True)
        for sheet in wb.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            headers = [str(h) for h in rows[0]]
            for row in rows[1:]:
                row_dict = {headers[i]: str(row[i]) if row[i] is not None else "" for i in range(len(headers))}
                if filter_dict:
                    skip = False
                    for k, v in filter_dict.items():
                        if row_dict.get(k) != v:
                            skip = True
                            break
                    if skip:
                        continue
                headings = {col: row_dict.get(col) for col in heading_columns} if heading_columns else None
                text = ', '.join(f"{k}: {v}" for k, v in row_dict.items())
                chunks.append((headings, text))
    else:
        raise ValueError("Unsupported table file type")
    return chunks

# ========== TOC PAGE FINDER ==========
def find_toc_page(pdf_path, max_search_pages=10):
    toc_keywords = [
        "table of contents",
        "contents",
        "index",
        "summary"
    ]
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = min(len(reader.pages), max_search_pages)
        for i in range(num_pages):
            page_text = reader.pages[i].extract_text()
            if page_text:
                page_text_lower = page_text.lower()
                if any(keyword in page_text_lower for keyword in toc_keywords):
                    return i
    return None

# ========== CHUNKING & SPLITTING ==========
def split_into_paragraphs(text):
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def extract_section_headings(toc_text):
    headings = []
    for line in toc_text.split('\n'):
        line = line.strip()
        if not line or line.isdigit() or line.upper() == "CONTENTS":
            continue
        match = re.match(r"^(.*?)(?:\s*[\.\s]+)?(\d+)$", line)
        if match:
            heading = match.group(1).strip()
            if heading:
                headings.append(heading)
    return headings

def chunk_by_paragraph(text):
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def chunk_by_headings(text, headings):
    """
    Splits text into sections by headings.
    Returns a list of (heading_dict, section_text).
    """
    normalized_headings = [h.strip().lower() for h in headings]
    lines = text.split('\n')
    sections = []
    current_section = []
    current_title = None

    for line in lines:
        normalized_line = line.strip().lower()
        if normalized_line in normalized_headings:
            if current_section and current_title:
                sections.append(({"Section": current_title}, "\n".join(current_section).strip()))
            current_title = line.strip()
            current_section = []
        else:
            current_section.append(line)
    if current_section and current_title:
        sections.append(({"Section": current_title}, "\n".join(current_section).strip()))
    return sections

def chunk_sections_to_paragraphs(sections):
    """
    Given a list of (heading_dict, section_text), split each section into paragraphs,
    assigning the heading to each paragraph.
    Returns a list of (heading_dict, paragraph) tuples.
    """
    result = []
    for heading_dict, section_text in sections:
        paragraphs = split_into_paragraphs(section_text)
        for para in paragraphs:
            if para.strip():
                result.append((heading_dict, para.strip()))
    return result

def re_chunk_sections(sections, max_tokens=8191):
    max_chars = max_tokens * 4
    new_sections = []
    for headings, section in sections:
        if len(section) <= max_chars:
            new_sections.append((headings, section))
        else:
            paragraphs = split_into_paragraphs(section)
            chunk = ""
            for para in paragraphs:
                if len(chunk) + len(para) + 2 <= max_chars:
                    chunk += para + "\n\n"
                else:
                    if chunk:
                        new_sections.append((headings, chunk.strip()))
                    if len(para) > max_chars:
                        sentences = split_into_sentences(para)
                        sent_chunk = ""
                        for sent in sentences:
                            if len(sent_chunk) + len(sent) + 1 <= max_chars:
                                sent_chunk += sent + " "
                            else:
                                new_sections.append((headings, sent_chunk.strip()))
                                sent_chunk = sent + " "
                        if sent_chunk:
                            new_sections.append((headings, sent_chunk.strip()))
                    else:
                        new_sections.append((headings, para.strip()))
                    chunk = ""
            if chunk:
                new_sections.append((headings, chunk.strip()))
    return new_sections

def re_chunk_paragraphs(paragraphs, max_tokens=8191):
    max_chars = max_tokens * 4
    new_sections = []
    chunk = ""
    for para in paragraphs:
        if len(chunk) + len(para) + 2 <= max_chars:
            chunk += para + "\n\n"
        else:
            if chunk:
                new_sections.append((None, chunk.strip()))
            if len(para) > max_chars:
                sentences = split_into_sentences(para)
                sent_chunk = ""
                for sent in sentences:
                    if len(sent_chunk) + len(sent) + 1 <= max_chars:
                        sent_chunk += sent + " "
                    else:
                        new_sections.append((None, sent_chunk.strip()))
                        sent_chunk = sent + " "
                if sent_chunk:
                    new_sections.append((None, sent_chunk.strip()))
            else:
                new_sections.append((None, para.strip()))
            chunk = ""
    if chunk:
        new_sections.append((None, chunk.strip()))
    return new_sections

# ========== EMBEDDING ==========
def get_embeddings(texts, client, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

# ========== OUTPUT PATHS ==========
def get_output_paths(source_path, embeddings_dir="./embeddings", chunks_dir="./chunks"):
    base = os.path.splitext(os.path.basename(source_path))[0]
    embeddings_path = os.path.join(embeddings_dir, f"{base}_embeddings.npy")
    chunks_path = os.path.join(chunks_dir, f"{base}_chunks.json")
    return embeddings_path, chunks_path

# ========== MAIN WORKFLOW ==========
def embedding_generator(
    FILE_PATH,
    BATCH_SIZE=20,
    TOC_PAGE=None,
    heading_columns=None,
    filter_dict=None
):
    EMBEDDINGS_PATH, CHUNKS_PATH = get_output_paths(FILE_PATH)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    ext = os.path.splitext(FILE_PATH)[1].lower()
    print(f"Processing file: {FILE_PATH}")

    patterns = [r'Daggerheart SRD', r'\bPage \d+\b', r'^\d+$']

    if ext in [".csv", ".xlsx"]:
        print("Extracting and chunking table (CSV/XLSX)...")
        raw_sections = extract_chunks_from_table(
            FILE_PATH,
            heading_columns=heading_columns,
            filter_dict=filter_dict
        )
        raw_sections = [(h, preprocess_text(s, patterns)) for h, s in raw_sections]
        sections = raw_sections

    elif ext == ".pdf":
        print("Searching for Table of Contents page...")
        toc_page = TOC_PAGE
        if toc_page is None:
            toc_page = find_toc_page(FILE_PATH, max_search_pages=10)
        if toc_page is not None:
            print(f"Found ToC on page {toc_page+1}")
            toc_text = extract_page_text(FILE_PATH, toc_page)
        else:
            print("ToC not found, defaulting to page 2.")
            toc_text = extract_page_text(FILE_PATH, 1)
        headings = extract_section_headings(toc_text)
        print(f"Section headings found in ToC: {headings}")
        full_text = extract_text_from_pdf(FILE_PATH)
        # Chunk by section heading
        section_chunks = chunk_by_headings(full_text, headings)
        # Now split each section into paragraphs, assigning the heading
        raw_sections = chunk_sections_to_paragraphs(section_chunks)
        # Preprocess after heading assignment
        raw_sections = [(h, preprocess_text(s, patterns)) for h, s in raw_sections]
        sections = re_chunk_sections(raw_sections, max_tokens=7000)

    elif ext == ".docx":
        print("Extracting and chunking DOCX by heading...")
        section_chunks = extract_sections_from_docx(FILE_PATH)
        raw_sections = chunk_sections_to_paragraphs(section_chunks)
        raw_sections = [(h, preprocess_text(s, patterns)) for h, s in raw_sections]
        sections = re_chunk_sections(raw_sections, max_tokens=7000)

    elif ext == ".txt":
        print("Extracting and chunking TXT...")
        full_text = extract_text_from_txt(FILE_PATH)
        paragraphs = chunk_by_paragraph(full_text)
        raw_sections = [(None, p) for p in paragraphs]
        raw_sections = [(h, preprocess_text(s, patterns)) for h, s in raw_sections]
        sections = re_chunk_sections(raw_sections, max_tokens=7000)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"Section count after chunking: {len(sections)}")
    for i, (headings, section) in enumerate(sections):
        print(f"Section {i}: {len(section) // 4} tokens (approx), Headings: {headings}")

    # 3. Prepare chunk dicts for output and embedding
    chunk_dicts = []
    for idx, (headings, text) in enumerate(sections):
        chunk_dicts.append({
            "index": idx,
            "headings": headings,
            "text": text
        })

    # 4. Generate embeddings in batches
    print("Generating embeddings...")
    openai_client = openai.OpenAI(api_key=config.openai_apikey)
    embeddings = []
    for i in range(0, len(chunk_dicts), BATCH_SIZE):
        batch = chunk_dicts[i:i+BATCH_SIZE]
        batch_texts = [c["text"] for c in batch]
        batch_embeddings = get_embeddings(batch_texts, openai_client)
        embeddings.append(batch_embeddings)
        print(f"Embedded batch {i//BATCH_SIZE + 1}/{(len(chunk_dicts)-1)//BATCH_SIZE + 1}")
    embeddings = np.vstack(embeddings)

    # 5. Save to disk
    print(f"Saving embeddings to {EMBEDDINGS_PATH} ...")
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saving chunks to {CHUNKS_PATH} ...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

    print("Done! You can now load these files in your Flask app.")

if __name__ == "__main__":
    # Example usage for CSV/XLSX:
    FILE_PATH = "./source/DH-SRD-May202025.pdf"  # or .csv, .pdf, .docx, .txt
    heading_columns = []  # or any columns you want for CSV/XLSX
    filter_dict = {}      # or {} for no filter
    embedding_generator(
        FILE_PATH,
        heading_columns=heading_columns,
        filter_dict=filter_dict
    )