from PyPDF2 import PdfReader
import textwrap

def load_and_chunk(uploaded_file, chunk_size=500):
    # Determine file type
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"
    else:  # .txt
        raw_text = uploaded_file.read().decode("utf-8")

    # Clean & chunk
    chunks = textwrap.wrap(raw_text, width=chunk_size, break_long_words=False, replace_whitespace=False)
    return chunks
