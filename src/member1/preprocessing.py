
from pathlib import Path
import re
from pypdf import PdfReader


# Find the main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output folders
INPUT_FOLDER = PROJECT_ROOT / "data" / "raw_docs"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed_docs"


def extract_text(pdf_path):
    """
    Extract text from every page of a PDF.
    """

    reader = PdfReader(str(pdf_path))

    pages_text = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    return "\n\n".join(pages_text)



def clean_text(text):
    """
    Clean extracted PDF text while preserving line
    breaks, headings and paragraph boundaries.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Process each line independently
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:

        # Remove unnecessary spaces and tabs
        line = re.sub(r"[ \t]+", " ", line)

        # Remove leading and trailing spaces
        line = line.strip()

        cleaned_lines.append(line)

    # Rejoin lines without destroying their boundaries
    text = "\n".join(cleaned_lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



def process_pdf(pdf_path):
    """
    Extract, clean and save one PDF.
    """

    print(f"\nProcessing: {pdf_path.name}")

    # Extract PDF text
    extracted_text = extract_text(pdf_path)

    if not extracted_text.strip():
        print("Warning: No extractable text found.")
        return

    # Clean extracted text
    cleaned_text = clean_text(extracted_text)

    # Create output filename
    output_filename = pdf_path.stem + ".txt"
    output_path = OUTPUT_FOLDER / output_filename

    # Save processed text
    output_path.write_text(cleaned_text, encoding="utf-8")

    print(f"Saved: {output_path.name}")
    print(f"Characters extracted: {len(extracted_text)}")
    print(f"Characters after cleaning: {len(cleaned_text)}")


def main():

    # Create output folder if it doesn't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    pdf_files = sorted(INPUT_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw_docs.")
        return

    print(f"Found {len(pdf_files)} PDF file(s).")

    for pdf_path in pdf_files:

        try:
            process_pdf(pdf_path)

        except Exception as error:
            print(f"Error processing {pdf_path.name}: {error}")

    print("\nPreprocessing completed!")


if __name__ == "__main__":
    main()
