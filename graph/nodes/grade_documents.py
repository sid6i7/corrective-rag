from typing import Any, Dict, Tuple
import concurrent.futures

from langchain.schema import Document

from utils import logger
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state(dict): The current graph state.
    Returns:
        state(dict): Filtered out irrelevant documents and updated web_search state
    """
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    def is_relevant(document:Document) -> Tuple[Document, bool]:
        grade = retrieval_grader.invoke(
            {"question": question, "document": document.page_content}
        )
        return document, grade.binary_score == "yes"
    

    filtered_docs = []
    web_search = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(is_relevant, document) for document in documents]
        for future in futures:
            document, is_document_relevant = future.result()
            if is_document_relevant:
                filtered_docs.append(document)
            else:
                web_search = True
                
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }