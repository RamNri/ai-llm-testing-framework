from simple_rag import (
    extract_text_from_pdf,
    create_chunks,
    create_vector_store
)

PDF_PATH = "data/documents/s22_manual.pdf"


if __name__ == "__main__":
    text = extract_text_from_pdf(PDF_PATH)

    chunks = create_chunks(text)

    print(f"Total chunks created: {len(chunks)}")

    create_vector_store(chunks)

    print("FAISS index created successfully.")