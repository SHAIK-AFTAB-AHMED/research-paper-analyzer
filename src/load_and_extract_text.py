import re
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extract text from every page of a PDF.

    Returns:
        str: Complete extracted text.
    """

    reader = PdfReader(str(pdf_path))

    full_text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text()

            if page_text:
                full_text_parts.append(page_text)

        except Exception as error:
            print(
                f"Warning: Could not extract text from page "
                f"{page_number}: {error}"
            )

    full_text = "\n".join(full_text_parts)

    return full_text


def extract_parent_title(full_text, parent_number):
    """
    Find the title of a main section.

    Example:
        3 Methodology

    For parent_number = "3",
    returns:
        "Methodology"
    """

    parent_title_match = re.search(
        rf"^{re.escape(parent_number)}\s+(.+)",
        full_text,
        re.MULTILINE
    )

    if parent_title_match:
        return parent_title_match.group(1).strip()

    return ""


def parse_sections(text):
    """
    Detect numbered research-paper headings.

    Supported examples:

        1 Introduction
        2 Related Work
        3.1 Dataset
        3.2 Model Architecture
        3.2.1 Attention Mechanism

    Returns a list of dictionaries containing section information
    and character positions.
    """

    heading_pattern = re.compile(
        r"^(\d+(?:\.\d+)*)\s+([A-Za-z][^\n]*)$",
        re.MULTILINE
    )

    sections = []

    for match in heading_pattern.finditer(text):

        number = match.group(1)
        title = match.group(2).strip()
        start_index = match.start()

        # Main section
        if "." not in number:

            sections.append(
                {
                    "section": f"{number} {title}",
                    "start": start_index
                }
            )

        # Subsection
        else:

            parent_number = number.split(".")[0]

            parent_title = extract_parent_title(
                text,
                parent_number
            )

            if parent_title:

                sections.append(
                    {
                        "section": parent_title,
                        "subsection": f"{number} {title}",
                        "start": start_index
                    }
                )

            else:

                sections.append(
                    {
                        "section": f"{number} {title}",
                        "start": start_index
                    }
                )

    return sections


def find_abstract(text):
    """
    Detect the Abstract section.
    """

    abstract_match = re.search(
        r"\bAbstract\b",
        text,
        re.IGNORECASE
    )

    if abstract_match:
        return {
            "section": "Abstract",
            "start": abstract_match.start()
        }

    return None


def extract_pdf_sections(full_text):
    """
    Extract section information from the complete research paper.
    """

    if not full_text or not full_text.strip():
        return []

    sections = parse_sections(full_text)

    abstract = find_abstract(full_text)

    if abstract:

        # Avoid adding duplicate Abstract entries
        abstract_exists = any(
            section.get("section", "").lower() == "abstract"
            for section in sections
        )

        if not abstract_exists:
            sections.insert(0, abstract)

    return sections