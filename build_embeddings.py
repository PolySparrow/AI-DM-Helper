import PyPDF2
import openai
import numpy as np
import json
import re
import unicodedata
import string
import config

# ========== PREPROCESSING ==========
def preprocess_text(text, remove_patterns=None):
    # Lowercase
    text = text.lower()
    # Remove non-printable characters
    text = ''.join(filter(lambda x: x in string.printable, text))
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove headers/footers
    if remove_patterns:
        for pat in remove_patterns:
            text = re.sub(pat, '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Standardize punctuation
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('–', '-')
    return text

# ========== PDF EXTRACTION ==========
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ========== PAGE EXTRACTION ==========
def extract_page_text(pdf_path, page_number):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        if page_number < len(reader.pages):
            return reader.pages[page_number].extract_text()
        else:
            return ""

# ========== PARAGRAPH AND SENTENCE SPLITTING ==========    
def split_into_paragraphs(text):
    # Split on double newlines or single newlines with indentation
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def split_into_sentences(text):
    # Naive sentence splitter (for English)
    return re.split(r'(?<=[.!?])\s+', text)

# ========== SECTION HEADINGS EXTRACTION ==========
def extract_section_headings(toc_text):
    headings = []
    for line in toc_text.split('\n'):
        line = line.strip()
        # Skip empty lines and lines that are just numbers or titles like "CONTENTS"
        if not line or line.isdigit() or line.upper() == "CONTENTS":
            continue
        # Match: everything before the last number (page number)
        match = re.match(r"^(.*?)(?:\s*[\.\s]+)?(\d+)$", line)
        if match:
            heading = match.group(1).strip()
            if heading:
                headings.append(heading)
    return headings

# ========== CHUNKING ==========
def chunk_by_paragraph(text):
    # Split on double newlines or single newlines with indentation
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paragraphs

def chunk_by_headings(text, headings):
    # Normalize headings for matching
    normalized_headings = [h.strip().lower() for h in headings]
    lines = text.split('\n')
    sections = []
    current_section = []
    current_title = None

    for line in lines:
        normalized_line = line.strip().lower()
        if normalized_line in normalized_headings:
            # Save the previous section
            if current_section and current_title:
                sections.append(f"{current_title}\n" + "\n".join(current_section).strip())
            current_title = line.strip()
            current_section = []
        else:
            current_section.append(line)
    # Add the last section
    if current_section and current_title:
        sections.append(f"{current_title}\n" + "\n".join(current_section).strip())
    return sections

def re_chunk_sections(sections, max_tokens=8191):
    max_chars = max_tokens * 4  # Approximate: 1 token ≈ 4 chars
    new_sections = []
    for section in sections:
        if len(section) <= max_chars:
            new_sections.append(section)
        else:
            # Split by paragraph
            paragraphs = split_into_paragraphs(section)
            chunk = ""
            for para in paragraphs:
                if len(chunk) + len(para) + 2 <= max_chars:
                    chunk += para + "\n\n"
                else:
                    if chunk:
                        new_sections.append(chunk.strip())
                    # If paragraph itself is too big, split by sentence
                    if len(para) > max_chars:
                        sentences = split_into_sentences(para)
                        sent_chunk = ""
                        for sent in sentences:
                            if len(sent_chunk) + len(sent) + 1 <= max_chars:
                                sent_chunk += sent + " "
                            else:
                                new_sections.append(sent_chunk.strip())
                                sent_chunk = sent + " "
                        if sent_chunk:
                            new_sections.append(sent_chunk.strip())
                    else:
                        new_sections.append(para.strip())
                    chunk = ""
            if chunk:
                new_sections.append(chunk.strip())
    return new_sections

# ========== EMBEDDING ==========
def get_embeddings(texts, client, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

# ========== MAIN WORKFLOW ==========
def embedding_genererator(PDF_PATH = "./source/DH-SRD-May202025.pdf",EMBEDDINGS_PATH = "./embeddings/rules_embeddings.npy",CHUNKS_PATH = "./chunks/rules.json",BATCH_SIZE = 20 ):
    # 1. Extract ToC from page 2 (page_number=1, since it's zero-indexed)
    toc_text = extract_page_text(PDF_PATH, 1)
    headings = extract_section_headings(toc_text)
    print(f"Section headings found in ToC: {headings}")

    # 2. Extract and preprocess full text
    print("Extracting and preprocessing full text from PDF...")
    full_text = extract_text_from_pdf(PDF_PATH)
    patterns = [r'Daggerheart SRD', r'\bPage \d+\b', r'^\d+$']
    #full_text = preprocess_text(full_text,patterns)  # You can add remove_patterns if you want

    # 3. Chunk by section headings
    print("Chunking by section headings...")
    sections = chunk_by_headings(full_text, headings)
    sections = [preprocess_text(s,patterns) for s in sections]
    print(f"Initial section count: {len(sections)}")

    # 4. Re-chunk sections to fit within token limits
    print("Re-chunking long sections...")
    sections = re_chunk_sections(sections, max_tokens=7000)  # Adjust as needed
    print(f"Section count after re-chunking: {len(sections)}")
    for i, section in enumerate(sections):
        print(f"Section {i}: {len(section) // 4} tokens (approx)")

    # 5. Generate embeddings in batches
    print("Generating embeddings...")
    openai_client = openai.OpenAI(api_key=config.openai_apikey)
    embeddings = []
    for i in range(0, len(sections), BATCH_SIZE):
        batch = sections[i:i+BATCH_SIZE]
        batch_embeddings = get_embeddings(batch, openai_client)
        embeddings.append(batch_embeddings)
        print(f"Embedded batch {i//BATCH_SIZE + 1}/{(len(sections)-1)//BATCH_SIZE + 1}")
    embeddings = np.vstack(embeddings)

    # 6. Save to disk
    print(f"Saving embeddings to {EMBEDDINGS_PATH} ...")
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saving chunks to {CHUNKS_PATH} ...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

    print("Done! You can now load these files in your Flask app.")

if __name__ == "__main__":
    embedding_genererator()