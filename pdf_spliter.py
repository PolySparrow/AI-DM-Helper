from PyPDF2 import PdfReader, PdfWriter
import logging_function
logger = logging_function.setup_logger()

def split_skip_pages(input_path, start_page, end_page,  output_path, skip_pages,):
    """
    Extract pages from start_page to end_page (inclusive), skipping pages in skip_pages list.

    Parameters:
    - input_path: path to input PDF
    - start_page: first page to include (1-based)
    - end_page: last page to include (1-based)
    - skip_pages: list of page numbers to skip (1-based)
    - output_path: path to save the output PDF
    """
    reader = PdfReader(input_path)
    writer = PdfWriter()

    start_idx = start_page - 1
    end_idx = end_page  # end_page inclusive in range

    for i in range(start_idx, min(end_idx, len(reader.pages))):
        # Page numbers are 1-based, so add 1 to i
        if (i + 1) not in skip_pages:
            writer.add_page(reader.pages[i])

    with open(output_path, "wb") as output_pdf:
        writer.write(output_pdf)

def expand_skip_pages(skip_list):
    """
    Expand a list of pages and page ranges into a flat list of page numbers.

    skip_list can contain integers (single pages) or tuples/lists of two integers (start, end).

    Example:
    [3, (5,7), 10] -> [3, 5, 6, 7, 10]
    """
    expanded = []
    for item in skip_list:
        if isinstance(item, int):
            expanded.append(item)
        elif (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and all(isinstance(x, int) for x in item)
        ):
            start, end = item
            expanded.extend(range(start, end + 1))
        else:
            raise ValueError("skip_list items must be int or tuple/list of two ints")
    return expanded


# Example usage with ranges:
skip_pages = [(73,101),(103,111),(119,135)]  # skip page 3, pages 5 to 7, and page 10
expanded_skip_pages = expand_skip_pages(skip_pages)

#print(f"Expanded skip pages: {expanded_skip_pages}")
# Example usage:
split_skip_pages("./source/DH-SRD-1.0-June-26-2025.pdf", 1, 135, "./source/core_rules.pdf", expanded_skip_pages)
split_skip_pages("./source/DH-SRD-1.0-June-26-2025.pdf", 73, 101, "./source/adversaries.pdf",[])
split_skip_pages("./source/DH-SRD-1.0-June-26-2025.pdf", 103, 111, "./source/environments.pdf",[])
split_skip_pages("./source/DH-SRD-1.0-June-26-2025.pdf", 119, 135, "./source/domain_card_reference.pdf",[])
###
# Example 1: No pages skipped (extract pages 1 to 5)
#split_skip_pages("input.pdf", 1, 5, [], "output_all_pages.pdf")

# Example 2: Skip specific pages (extract pages 1 to 10, skip pages 3, 6, 9)
#split_skip_pages("input.pdf", 1, 10, [3, 6, 9], "output_skip_some.pdf")

# Example 3: Extract a single page (page 7)
#split_skip_pages("input.pdf", 7, 7, [], "output_single_page.pdf")

# Example 4: Skip all pages (pages 1 to 5 skipped, results in empty PDF)
#split_skip_pages("input.pdf", 1, 5, [1, 2, 3, 4, 5], "output_empty.pdf")

# Example 5: Extract from page 5 to the end, skipping page 8
#reader = PdfReader("input.pdf")
#total_pages = len(reader.pages)
#split_skip_pages("input.pdf", 5, total_pages, [8], "output_from_5_skip_8.pdf")