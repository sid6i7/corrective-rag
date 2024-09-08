from typing import Any, Dict

from utils import logger
from graph.state import GraphState


def retrieve(state: GraphState) -> Dict[str, Any]:
    logger.info("---RETRIEVE---")
    question = state["question"]
    retriever = state["retriever"]
    documents = retriever.invoke(question)
    return {
        "documents": documents,
        "question": question
    }