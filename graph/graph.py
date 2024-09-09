from typing import Set

from langgraph.graph import END, StateGraph

from utils import logger
from graph.constants import (
    RETRIEVE,
    GRADE_DOCUMENTS,
    GENERATE,
    WEB_SEARCH
)
from graph.chains.answer_grader import answer_grader_chain
from graph.chains.hallucination_grader import hallucination_grader_chain
from graph.chains.keyword_extractor import DocumentKeywords, keyword_extractor_chain
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.ingest import RAGVectorStore


def is_answer_grounded_in_documents(state: GraphState) -> str:
    logger.info("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    answer = state["answer"]

    is_grounded = hallucination_grader_chain.invoke(
        {"documents": documents, "answer": answer}
    )
    if is_grounded:
        logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        logger.info("---CHECKING IF LLM GENERATION ANSWERED THE QUESTION---")
        is_answered = answer_grader_chain.invoke(
            {"question": question, "answer": answer}
        )
        if is_answered:
            logger.info("---DECISION: GENERATION ANSWERED THE QUESTION---")
            return "useful"
        else:
            logger.info("---DECISION: GENERATION DOES NOT ADDRESS THE QUESTION---")
            return "not_useful"
    else:
        logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-RUNNING THE CHAIN")
        return "not_supported"


def decide_to_generate(state):
    if state["web_search"]:
        logger.info(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION"
        )
        return WEB_SEARCH
    else:
        logger.info("---DECISION: GENERATE---")
        return GENERATE


def decide_to_route(state: GraphState):
    question:str = state["question"]
    keywords_in_question:DocumentKeywords = keyword_extractor_chain.invoke({"document": question})
    keywords_in_question:Set[str] = set(keywords_in_question.keywords)
    retriever:RAGVectorStore = state["retriever"]
    keywords_in_db:Set[str] = retriever.get_keywords_in_vector_store()
    #TODO: find a more scalable solution when amount of keywords in db grows exponentially
    if keywords_in_db.isdisjoint(keywords_in_question):
        logger.info("---ROUTE DECISION: WEB SEARCH")
        return WEB_SEARCH
    logger.info("---ROUTE DECISION: VECTOR STORE")
    return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

workflow.set_conditional_entry_point(decide_to_route)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate)
workflow.add_conditional_edges(
    GENERATE,
    is_answer_grounded_in_documents,
    path_map={
        "useful": END,
        "not_useful": WEB_SEARCH,
        "not_supported": GENERATE
    }
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()