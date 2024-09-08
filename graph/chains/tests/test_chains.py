from typing import List
from pprint import pprint

from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.generation import generation_chain
from graph.ingest import RAGVectorStore


load_dotenv()
retriever:VectorStoreRetriever = RAGVectorStore.get_retriever(
                                            collection_name="rag-chroma",
                                            persist_directory="./.chroma"
                                        )


def test_retrieval_grader_answer_yes() -> None:
    question = "prompt engineering"
    docs:List[Document] = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "how to make pizza"
    docs:List[Document] = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "how to reduce hallucinations in LLMs?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)