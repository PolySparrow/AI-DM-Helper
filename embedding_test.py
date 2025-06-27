import PyPDF2
import openai
import numpy as np
import faiss
import re
import config
import string
import unicodedata

# Set up OpenAI client
client = openai.OpenAI(api_key=config.openai_apikey)

pdf_path = "DH-SRD-May202025.pdf"

def extract_page_text(pdf_path, page_number):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        if page_number < len(reader.pages):
            return reader.pages[page_number].extract_text()
        else:
            return ""

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

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

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

def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return np.array([d.embedding for d in response.data], dtype="float32")

def split_into_paragraphs(text):
    # Split on double newlines or single newlines with indentation
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def split_into_sentences(text):
    # Naive sentence splitter (for English)
    return re.split(r'(?<=[.!?])\s+', text)

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

def write_chunks_to_file(chunks, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(chunk.strip())
            f.write("\n\n---\n\n")  # Separator between chunks
    print(f"Wrote {len(chunks)} chunks to {filename}")

# --- Main workflow ---

# 1. Extract ToC from page 2 (page_number=1, since it's zero-indexed)
toc_text = extract_page_text(pdf_path, 1)
headings = extract_section_headings(toc_text)
#print("Section headings found in ToC:", headings)

# 2. Extract full text
full_text = extract_text_from_pdf(pdf_path)
patterns = [r'Daggerheart SRD', r'\bPage \d+\b', r'^\d+$']
clean_text = preprocess_text(full_text, remove_patterns=patterns)


# 3. Chunk by headings
sections = chunk_by_headings(full_text, headings)
sections = [preprocess_text(s, remove_patterns=patterns) for s in sections]
for i, section in enumerate(sections):
    print(f"Section {i}: {len(section) // 4} tokens (approx)")
# 3. Re-chunk sections to fit within token limits
sections = re_chunk_sections(sections, max_tokens=7000)  # Adjust max_tokens as needed
print("\nAfter re-chunking:")
for i, section in enumerate(sections):
    print(f"Section {i}: {len(section) // 4} tokens (approx)")
print(f"Extracted {len(sections)} sections from the PDF.")
# Suppose 'sections' is your list of chunks
write_chunks_to_file(sections, "daggerheart_chunks.txt")

# 4. Generate embeddings in batches
embeddings = []
batch_size = 10
for i in range(0, len(sections), batch_size):
    batch = sections[i:i+batch_size]
    batch_embeddings = get_embeddings(batch)
    embeddings.append(batch_embeddings)
embeddings = np.vstack(embeddings)

# 5. Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 6. Query function
def search(query, k=3):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    print("Top relevant sections:")
    for i in indices[0]:
        print("\n---\n", sections[i][:])  # Print first 1000 chars for brevity

# Example query
search("when does the DM generate fear?", k=5)