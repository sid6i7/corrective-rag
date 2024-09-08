from typing import Any, Dict

from utils import logger
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    answer = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "answer": answer}