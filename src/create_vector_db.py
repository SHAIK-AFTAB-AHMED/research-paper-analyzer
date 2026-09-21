from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def create_vector_db(text, embedder):
    """
    Create a FAISS vector database from the research paper text.

    The paper is split into smaller overlapping chunks before
    generating embeddings.
    """

    if not text or not text.strip():
        raise ValueError("Cannot create vector database from empty text.")

    # Create a LangChain document
    document = Document(page_content=text)

    # Split the research paper into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = splitter.split_documents([document])

    print(f"Created {len(documents)} text chunks.")

    if not documents:
        raise ValueError("No text chunks were created from the research paper.")

    # Create FAISS vector database
    vector_db = FAISS.from_documents(
        documents,
        embedding=embedder
    )

    # Save the vector database locally
    vector_db.save_local("research_paper_vector_db")

    print("FAISS vector database created successfully.")

    return vector_db