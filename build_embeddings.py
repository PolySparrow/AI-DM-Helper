import os
import csv
import json
import re
import unicodedata
import string
import numpy as np
from typing import Optional

import PyPDF2
import pdfplumber
from docx import Document
import openpyxl
from sentence_transformers import SentenceTransformer

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
# ========== TEXT EXTRACTION FROM TXT ==========
def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()
# ========== PDF PAGE EXTRACTION ==========
def extract_page_text(pdf_path: str, page_number: int) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        if 0 <= page_number < len(reader.pages):
            return reader.pages[page_number].extract_text() or ""
        else:
            return ""
        
# ========== PDFPLUMBER CHUNKING FOR CORE RULES ==========
def is_heading(word, font_size, min_font_size, min_len=4, max_words=6):
    return (
        word.isupper() and
        font_size >= min_font_size and
        min_len <= len(word) <= 40 and
        1 <= len(word.split()) <= max_words and
        not re.search(r'[^\w\s]', word)
    )

def chunk_pdf_with_fontsize(pdf_path):
    import pdfplumber
    import re

    def is_heading(word, font_size, target_font, min_len=4, max_words=6):
        return (
            word.isupper() and
            font_size == target_font and
            min_len <= len(word) <= 40 and
            1 <= len(word.split()) <= max_words and
            not re.search(r'[^\w\s]', word)
        )

    with pdfplumber.open(pdf_path) as pdf:
        # 1. Find the largest font sizes (rounded), skipping the first page
        font_sizes = []
        for page in pdf.pages[1:]:
            for char in page.chars:
                font_sizes.append(round(char["size"]))
        font_sizes = sorted(set(font_sizes), reverse=True)
        if len(font_sizes) < 5:
            raise ValueError("Not enough font size variation to detect 4 heading layers.")
        # Skip the largest (title) font size
        main_section_font = font_sizes[1]
        subsection_font = font_sizes[2]
        subsubsection_font = font_sizes[3]
        subsubsubsection_font = font_sizes[4]
        print("Unique font sizes (descending, skipping title):", font_sizes)
        print(f"Detected main section font size: {main_section_font}")
        print(f"Detected subsection font size: {subsection_font}")
        print(f"Detected subsubsection font size: {subsubsection_font}")
        print(f"Detected subsubsubsection font size: {subsubsubsection_font}")

        # 2. Parse and chunk
        chunks = []
        current_section = None
        current_subsection = None
        current_subsubsection = None
        current_subsubsubsection = None
        current_chunk_lines = []
        current_page = 2  # Since we skip the first page

        # For collecting the first 5 found headings at each layer
        main_section_headings = []
        subsection_headings = []
        subsubsection_headings = []
        subsubsubsection_headings = []

        for page_num, page in enumerate(pdf.pages[1:], start=1):
            words = page.extract_words(extra_attrs=["size"])
            for w in words:
                word_text = w["text"].strip()
                font_size = round(w["size"])
                # Main section
                if is_heading(word_text, font_size, main_section_font):
                    if len(main_section_headings) < 5:
                        main_section_headings.append((word_text, page_num + 1))
                    if current_chunk_lines and current_section:
                        chunks.append({
                            "section": current_section,
                            "subsection": current_subsection,
                            "subsubsection": current_subsubsection,
                            "subsubsubsection": current_subsubsubsection,
                            "page": current_page,
                            "text": "\n".join(current_chunk_lines).strip()
                        })
                    current_section = word_text
                    current_subsection = None
                    current_subsubsection = None
                    current_subsubsubsection = None
                    current_chunk_lines = []
                    current_page = page_num + 1
                # Subsection
                elif is_heading(word_text, font_size, subsection_font):
                    if len(subsection_headings) < 5:
                        subsection_headings.append((word_text, page_num + 1))
                    if current_chunk_lines and current_section:
                        chunks.append({
                            "section": current_section,
                            "subsection": current_subsection,
                            "subsubsection": current_subsubsection,
                            "subsubsubsection": current_subsubsubsection,
                            "page": current_page,
                            "text": "\n".join(current_chunk_lines).strip()
                        })
                    current_subsection = word_text
                    current_subsubsection = None
                    current_subsubsubsection = None
                    current_chunk_lines = []
                    current_page = page_num + 1
                # Subsubsection
                elif is_heading(word_text, font_size, subsubsection_font):
                    if len(subsubsection_headings) < 5:
                        subsubsection_headings.append((word_text, page_num + 1))
                    if current_chunk_lines and current_section:
                        chunks.append({
                            "section": current_section,
                            "subsection": current_subsection,
                            "subsubsection": current_subsubsection,
                            "subsubsubsection": current_subsubsubsection,
                            "page": current_page,
                            "text": "\n".join(current_chunk_lines).strip()
                        })
                    current_subsubsection = word_text
                    current_subsubsubsection = None
                    current_chunk_lines = []
                    current_page = page_num + 1
                # Subsubsubsection
                elif is_heading(word_text, font_size, subsubsubsection_font):
                    if len(subsubsubsection_headings) < 5:
                        subsubsubsection_headings.append((word_text, page_num + 1))
                    if current_chunk_lines and current_section:
                        chunks.append({
                            "section": current_section,
                            "subsection": current_subsection,
                            "subsubsection": current_subsubsection,
                            "subsubsubsection": current_subsubsubsection,
                            "page": current_page,
                            "text": "\n".join(current_chunk_lines).strip()
                        })
                    current_subsubsubsection = word_text
                    current_chunk_lines = []
                    current_page = page_num + 1
                else:
                    current_chunk_lines.append(word_text)
        if current_chunk_lines and current_section:
            chunks.append({
                "section": current_section,
                "subsection": current_subsection,
                "subsubsection": current_subsubsection,
                "subsubsubsection": current_subsubsubsection,
                "page": current_page,
                "text": "\n".join(current_chunk_lines).strip()
            })

    # Print the first 5 main section and subsection headings found
    print("\nFirst 5 main section headings and their pages:")
    for heading, page in main_section_headings:
        print(f"  '{heading}' on page {page}")
    print("\nFirst 5 subsection headings and their pages:")
    for heading, page in subsection_headings:
        print(f"  '{heading}' on page {page}")
    print("\nFirst 5 subsubsection headings and their pages:")
    for heading, page in subsubsection_headings:
        print(f"  '{heading}' on page {page}")
    print("\nFirst 5 subsubsubsection headings and their pages:")
    for heading, page in subsubsubsection_headings:
        print(f"  '{heading}' on page {page}")

    return chunks
# ========== STAT BLOCK CHUNKING FOR ADVERSARIES.PDF ==========
def chunk_stat_blocks_from_pdf(pdf_path: str):
    stat_blocks = []
    name_pattern = re.compile(r'^[A-Z][A-Z\s\-]+[A-Z]$')
    skip_words = {"FEATURES"}

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            lines = text.split('\n')
            current_block = []
            current_name = None
            for line in lines:
                line_stripped = line.strip()
                if name_pattern.match(line_stripped) and line_stripped not in skip_words:
                    if current_block and current_name:
                        stat_blocks.append({
                            "name": current_name,
                            "page": page_num + 1,
                            "text": "\n".join(current_block).strip()
                        })
                    current_name = line_stripped
                    current_block = [line_stripped]
                else:
                    if current_block is not None:
                        current_block.append(line)
            if current_block and current_name:
                stat_blocks.append({
                    "name": current_name,
                    "page": page_num + 1,
                    "text": "\n".join(current_block).strip()
                })
    return stat_blocks

# ========== DOMAIN CARD CHUNKING FROM PDF ==========
def chunk_domain_cards_from_pdf(pdf_path: str):
    card_blocks = []
    domain_pattern = re.compile(r'^[A-Z\s]+DOMAIN$')
    card_pattern = re.compile(r'^(■\s*)?([A-Z][A-Z\s\-]+[A-Z])$')
    skip_words = {"APPENDIX", "DOMAIN CARD REFERENCE"}

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            lines = text.split('\n')
            current_domain = None
            current_card = None
            current_block = []
            for line in lines:
                line_stripped = line.strip()
                # Detect domain
                if domain_pattern.match(line_stripped):
                    current_domain = line_stripped
                    continue
                # Detect card/spell name
                if card_pattern.match(line_stripped) and line_stripped not in skip_words:
                    # Save previous card
                    if current_block and current_card:
                        card_blocks.append({
                            "domain": current_domain,
                            "name": current_card,
                            "page": page_num + 1,
                            "text": "\n".join(current_block).strip()
                        })
                    current_card = card_pattern.match(line_stripped).group(2)
                    current_block = [line_stripped]
                else:
                    if current_block is not None:
                        current_block.append(line)
            # Save last card on the page
            if current_block and current_card:
                card_blocks.append({
                    "domain": current_domain,
                    "name": current_card,
                    "page": page_num + 1,
                    "text": "\n".join(current_block).strip()
                })
    return card_blocks

# ========== PDF SECTION CHUNKING WITH PAGE TAGS ==========
def extract_sections_from_pdf(pdf_path: str, headings: list):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        heading_pages = {}
        normalized_headings = [h.strip().lower() for h in headings]
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            lines = text.split('\n')
            for line in lines:
                norm_line = line.strip().lower()
                if norm_line in normalized_headings:
                    idx = normalized_headings.index(norm_line)
                    heading_pages[headings[idx]] = page_num + 1

        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        sections = []
        normalized_headings = [h.strip().lower() for h in headings]
        lines = full_text.split('\n')
        current_section = []
        current_title = None
        for line in lines:
            norm_line = line.strip().lower()
            if norm_line in normalized_headings:
                if current_section and current_title:
                    page_num = heading_pages.get(current_title, None)
                    sections.append(({"Section": current_title, "Page": page_num}, "\n".join(current_section).strip()))
                current_title = line.strip()
                current_section = []
            else:
                current_section.append(line)
        if current_section and current_title:
            page_num = heading_pages.get(current_title, None)
            sections.append(({"Section": current_title, "Page": page_num}, "\n".join(current_section).strip()))
        return sections

# ========== DOCX SECTION CHUNKING ==========
def extract_sections_from_docx(docx_path: str):
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

# ========== EMBEDDING ==========
def get_embeddings(texts, embedder):
    return embedder.encode(texts, normalize_embeddings=True)

# ========== OUTPUT PATHS ==========
def get_output_paths(source_path, embeddings_dir="./embeddings", chunks_dir="./chunks"):
    base = os.path.splitext(os.path.basename(source_path))[0]
    embeddings_path = os.path.join(embeddings_dir, f"{base}_embeddings.npy")
    chunks_path = os.path.join(chunks_dir, f"{base}_chunks.json")
    return embeddings_path, chunks_path

# ========== MAIN WORKFLOW ==========
def embedding_generator(
    FILE_PATH,
    embedder,
    BATCH_SIZE=20,
    TOC_PAGE=None,
    heading_columns=None,
    filter_dict=None
):
    EMBEDDINGS_PATH, CHUNKS_PATH = get_output_paths(FILE_PATH)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)

    ext = os.path.splitext(FILE_PATH)[1].lower()
    base = os.path.splitext(os.path.basename(FILE_PATH))[0]
    print(f"Processing file: {FILE_PATH}")

    patterns = [r'Daggerheart SRD', r'\bPage \d+\b', r'^\d+$']

    # Special-case for core rules PDF using pdfplumber
    if base == "core_rules":
        print("Special font-size-based chunking for core rules PDF ...")
        raw_chunks = chunk_pdf_with_fontsize(FILE_PATH)
        chunk_dicts = []
        for idx, chunk in enumerate(raw_chunks):
            chunk_dicts.append({
                "index": idx,
                "section": chunk["section"],
                "subsection": chunk["subsection"],
                "page": chunk["page"],
                "text": preprocess_text(chunk["text"], patterns)
            })
        texts = [c["text"] for c in chunk_dicts]

    # Special-case for adversaries.pdf
    if base.lower() == "adversaries" or base.lower() == "environments":
        print("Special stat block chunking for adversaries.pdf ...")
        stat_blocks = chunk_stat_blocks_from_pdf(FILE_PATH)
        chunk_dicts = []
        for idx, block in enumerate(stat_blocks):
            chunk_dicts.append({
                "index": idx,
                "name": block["name"],
                "page": block["page"],
                "text": preprocess_text(block["text"], patterns)
            })
        texts = [c["text"] for c in chunk_dicts]
    elif base.lower() == "domain_card_reference":
        print("Special domain card chunking for domain_card_reference.pdf ...")
        card_blocks = chunk_domain_cards_from_pdf(FILE_PATH)
        chunk_dicts = []
        for idx, block in enumerate(card_blocks):
            chunk_dicts.append({
                "index": idx,
                "domain": block["domain"],
                "name": block["name"],
                "page": block["page"],
                "text": preprocess_text(block["text"], patterns)
            })
        texts = [c["text"] for c in chunk_dicts]
    elif ext in [".csv", ".xlsx"]:
        print("Extracting and chunking table (CSV/XLSX)...")
        raw_sections = extract_chunks_from_table(
            FILE_PATH,
            heading_columns=heading_columns,
            filter_dict=filter_dict
        )
        raw_sections = [(h, preprocess_text(s, patterns)) for h, s in raw_sections]
        chunk_dicts = []
        for idx, (headings, text) in enumerate(raw_sections):
            chunk_dicts.append({
                "index": idx,
                "headings": headings,
                "text": text
            })
        texts = [c["text"] for c in chunk_dicts]
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
        section_chunks = extract_sections_from_pdf(FILE_PATH, headings)
        raw_sections = [(h, preprocess_text(s, patterns)) for h, s in section_chunks]
        chunk_dicts = []
        for idx, (headings, text) in enumerate(raw_sections):
            chunk_dicts.append({
                "index": idx,
                "headings": headings,
                "text": text
            })
        texts = [c["text"] for c in chunk_dicts]
    elif ext == ".docx":
        print("Extracting and chunking DOCX by heading...")
        section_chunks = extract_sections_from_docx(FILE_PATH)
        raw_sections = [(h, preprocess_text(s, patterns)) for h, s in section_chunks]
        chunk_dicts = []
        for idx, (headings, text) in enumerate(raw_sections):
            chunk_dicts.append({
                "index": idx,
                "headings": headings,
                "text": text
            })
        texts = [c["text"] for c in chunk_dicts]
    elif ext == ".txt":
        print("Extracting and chunking TXT...")
        full_text = extract_text_from_txt(FILE_PATH)
        chunk_dicts = [{
            "index": 0,
            "headings": None,
            "text": preprocess_text(full_text, patterns)
        }]
        texts = [chunk_dicts[0]["text"]]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"Chunk count: {len(chunk_dicts)}")
    for i, chunk in enumerate(chunk_dicts):
        print(f"Chunk {i}: {len(chunk['text']) // 4} tokens (approx), Meta: {chunk.get('headings', chunk.get('name'))}")

    # Generate embeddings in batches
    print("Generating embeddings...")
    embeddings = []
    for i in range(0, len(chunk_dicts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_embeddings = get_embeddings(batch_texts, embedder)
        embeddings.append(batch_embeddings)
        print(f"Embedded batch {i//BATCH_SIZE + 1}/{(len(chunk_dicts)-1)//BATCH_SIZE + 1}")
    embeddings = np.vstack(embeddings)

    # Save to disk
    print(f"Saving embeddings to {EMBEDDINGS_PATH} ...")
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saving chunks to {CHUNKS_PATH} ...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)

    print("Done! You can now load these files in your search app.")

if __name__ == "__main__":
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Example: process all files in ./source
    source_dir = "./source"
    skip_files = {"DH-SRD-1.0-June-26-2025.pdf"}

    for fname in os.listdir(source_dir):
        if not fname.lower().endswith((".pdf", ".docx", ".txt", ".csv", ".xlsx")):
            continue
        if fname in skip_files:
            print(f"Skipping {fname}")
            continue
        FILE_PATH = os.path.join(source_dir, fname)
        embedding_generator(
            FILE_PATH,
            embedder=embedder
        )