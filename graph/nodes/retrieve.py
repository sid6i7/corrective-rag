from typing import Any, Dict, List

from langchain.schema import Document

from utils import logger
from graph.state import GraphState
from graph.ingest import RAGVectorStore


def retrieve(state: GraphState) -> Dict[str, Any]:
    #TODO: add metadata filtering
    logger.info("---RETRIEVE---")
    question:str = state["question"]
    retriever:RAGVectorStore = state["retriever"]
    documents:List[Document] = retriever.get_retriever().invoke(question)
    return {
        "documents": documents,
        "question": question
    }