def chunk_text(pages, chunk_size=500):

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        for i in range(0, len(text), chunk_size):

            chunks.append({
                "text": text[i:i + chunk_size],
                "page": page_number
            })

    return chunks