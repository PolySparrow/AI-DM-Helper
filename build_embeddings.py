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
from collections import defaultdict

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

# ========== TXT EXTRACTION ==========
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

# ========== TOC PAGE FINDER ==========
def find_toc_page(pdf_path, max_search_pages=10):
    toc_keywords = ["table of contents", "contents", "index", "summary"]
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

# ========== SUBSECTION EXTRACTION FROM SECTION ==========
def extract_subsections_from_section(pdf_path, start_page, end_page, section_heading, next_section_heading=None, min_font=14, max_font=20):
    """
    Extracts all ALL CAPS lines with font size in [min_font, max_font] from the given page range,
    but stops if the next section heading is encountered.
    """
    def is_subsection(line, font_size, min_font=13, max_font=20, min_len=1, max_words=8):
        return (
            line.isupper() and
            min_font <= font_size <= max_font and
            min_len <= len(line) <= 60 and
            1 <= len(line.split()) <= max_words and
            not re.search(r'[^\w\s]', line)
        )

    subsections = []
    seen = set()
    found_section = False
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        start_page = max(1, min(start_page, num_pages))
        end_page = max(1, min(end_page, num_pages))
        for page_num in range(start_page - 1, end_page):
            page = pdf.pages[page_num]
            lines_dict = defaultdict(list)
            for w in page.extract_words(extra_attrs=["size", "y0"]):
                lines_dict[round(w["top"])].append(w)
            for y0 in sorted(lines_dict.keys()):
                words = lines_dict[y0]
                line_text = " ".join(w["text"].strip() for w in words)
                font_size = max(round(w["size"]) for w in words)
                # If we see the next section heading, stop collecting
                if next_section_heading and line_text.strip().lower() == next_section_heading.strip().lower():
                    return subsections
                # If we see the current section heading, start collecting
                if not found_section and line_text.strip().lower() == section_heading.strip().lower():
                    found_section = True
                    continue
                if found_section and is_subsection(line_text, font_size, min_font=min_font, max_font=max_font):
                    if line_text not in seen:
                        subsections.append(line_text)
                        seen.add(line_text)
    return subsections

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

    # Special-case for core rules PDF using ToC headings and font size 14 subsections as metadata
    if base == "core_rules":
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
        chunk_dicts = []
        num_sections = len(section_chunks)
        for idx, (headings_dict, text) in enumerate(section_chunks):
            start_page = headings_dict.get("Page", 1)
            section_heading = headings_dict.get("Section", "")
            # Get the next section heading if available
            if idx + 1 < len(section_chunks):
                next_section_heading = section_chunks[idx + 1][0].get("Section", None)
                next_page = section_chunks[idx + 1][0].get("Page")
                if next_page is None or start_page is None:
                    end_page = start_page if start_page is not None else 1
                elif next_page <= start_page:
                    end_page = start_page
                else:
                    end_page = next_page - 1
            else:
                next_section_heading = None
                end_page = start_page

            if start_page is None:
                start_page = 1
            if end_page is None or end_page < start_page:
                end_page = start_page

            subsections = extract_subsections_from_section(
                FILE_PATH, start_page, end_page, section_heading, next_section_heading, min_font=14, max_font=20
            )
            chunk_dicts.append({
                "index": idx,
                "headings": headings_dict,
                "subsections": subsections,
                "text": preprocess_text(text, patterns)
            })
        texts = [c["text"] for c in chunk_dicts]
    elif base.lower() == "adversaries" or base.lower() == "environments":
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
        print(
            f"Chunk {i}: {len(chunk['text']) // 4} tokens (approx), "
            f"Meta: {chunk.get('headings', chunk.get('name'))}, "
            f"Subsections: {chunk.get('subsections', [])}"
        )

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
    skip_files = set('DH-SRD-1.0-June-26-2025.pdf')  # Add any files you want to skip

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