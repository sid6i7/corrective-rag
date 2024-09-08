from typing import List

from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.schema.runnable import RunnableSequence

from graph.constants import DEFAULT_COMPLETIONS_MODEL


llm = ChatOpenAI(model=DEFAULT_COMPLETIONS_MODEL, temperature=0)


class DocumentKeywords(BaseModel):
    """All the keywords present and relevant to a given document"""
    keywords: List[str] = Field(
        ...,
        description="List of all the relevant keywords that are present in a document. Ignoring any irrelevant words that don't provide meaningful information."
    )


structured_llm_keyword_extractor = llm.with_structured_output(DocumentKeywords)

system_prompt = """You are an expert an extracting keywords from a given document. Keywords can have either single word or two words at maximum."""
extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Here is the document:\n\n{document}")
    ]
)

keyword_extractor_chain:RunnableSequence = extraction_prompt | structured_llm_keyword_extractor