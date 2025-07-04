import pdfplumber
from logging_function import setup_logger
import logging
setup_logger(app_name="AI_DM_RAG")  # or whatever app name you want
logger = logging.getLogger(__name__)

def get_font_sizes(pdf_path):
    font_sizes = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for char in page.chars:
                font_sizes.append(round(char["size"]))
    font_sizes = sorted(set(font_sizes), reverse=True)
    print("Unique font sizes (descending):", font_sizes)
    return font_sizes

font_sizes = get_font_sizes("./source/DH-SRD-1.0-June-26-2025.pdf")
# Output: e.g., [24.0, 18.0, 14.0, 12.0, 10.0]
print(f"Total unique font sizes: {len(font_sizes)}"
      f" (first 5: {font_sizes[:]})")