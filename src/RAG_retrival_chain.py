from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def get_qa_chain(vectordb, llm):
    """
    Create a RetrievalQA chain using the FAISS vector database.

    The model is instructed to answer only from the retrieved
    research-paper context.
    """

    if vectordb is None:
        raise ValueError("Vector database is not available.")

    if llm is None:
        raise ValueError("LLM is not available.")

    # Create retriever
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    prompt_template = """
You are an assistant that answers questions about a research paper.

You MUST answer the question only using the information provided
in the context below.

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Instructions:

- If the answer is present in the context, answer clearly and concisely.
- If the answer cannot be found in the context, reply exactly:

I don't know.

Do not invent information.
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=[
            "context",
            "question"
        ]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    return chain