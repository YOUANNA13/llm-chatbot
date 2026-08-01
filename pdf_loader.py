from pypdf import PdfReader


def load_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages):

        page_text = page.extract_text()

        if page_text:

            pages.append({
                "page": page_number + 1,
                "text": page_text
            })

    return pages