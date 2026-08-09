def chunk_text(pages, filename, chunk_size=500):
    """
    Split extracted page text into chunks.

    Each chunk now carries the source filename, so chunks from multiple
    documents can live in the same vector store and still be traced back
    to the right file.
    """

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        for i in range(0, len(text), chunk_size):

            chunks.append({
                "text": text[i:i + chunk_size],
                "page": page_number,
                "filename": filename
            })

    return chunks