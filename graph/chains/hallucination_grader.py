from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

from graph.constants import DEFAULT_COMPLETIONS_MODEL


llm = ChatOpenAI(model=DEFAULT_COMPLETIONS_MODEL, temperature=0)


class GradeHallucination(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(
        description="Answer is grounded in the facts, True or False"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.\n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {answer}")
    ]
)


hallucination_grader:RunnableSequence = hallucination_prompt | structured_llm_grader
