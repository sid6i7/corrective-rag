from typing import List
from pprint import pprint

from dotenv import load_dotenv
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.hallucination_grader import GradeHallucination, hallucination_grader_chain
from graph.chains.keyword_extractor import DocumentKeywords, keyword_extractor_chain
from graph.chains.generation import generation_chain
from graph.ingest import RAGVectorStore
from constants import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIRECTORY


load_dotenv()
retriever:VectorStoreRetriever = RAGVectorStore.get_retriever(
                                            collection_name=DEFAULT_COLLECTION_NAME,
                                            persist_directory=DEFAULT_PERSIST_DIRECTORY
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


def test_hallucination_grader_answer_yes() -> None:
    question = "prompt engineering"
    docs = retriever.invoke(question)

    answer:str = generation_chain.invoke({"context": docs, "question": question})
    res:GradeHallucination = hallucination_grader_chain.invoke(
        {"documents": docs, "answer": answer}
    )
    assert res.binary_score == True


def test_hallucination_grader_answer_no() -> None:
    question = "prompt engineering"
    docs = retriever.invoke(question)

    res:GradeHallucination = hallucination_grader_chain.invoke(
        {
            "documents": docs,
            "answer": "In order to make pizza, we first need to start with the dough"
        }
    )
    assert res.binary_score == False


def test_keyword_extractor_chain() -> None:
    document = "Prompt engineering is the process of structuring an instruction that can be interpreted and understood by a generative AI model."
    expected_keywords = ["prompt engineering", "process", "structuring", "instruction", "generative ai", "model"]
    res:DocumentKeywords = keyword_extractor_chain.invoke(
        {"document": document}
    )
    keywords = res.keywords
    print(keywords)
    matched_keywords = [keyword for keyword in keywords if keyword.lower() in expected_keywords]
    assert len(matched_keywords) >= len(expected_keywords)//2