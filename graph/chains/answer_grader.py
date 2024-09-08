from langchain.schema.runnable import RunnableSequence
from langchain.pydantic_v1 import Field, BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph.constants import DEFAULT_COMPLETIONS_MODEL


llm = ChatOpenAI(model=DEFAULT_COMPLETIONS_MODEL, temperature=0)


class GradeAnswer(BaseModel):
    """To verify if the answer string actually answers the question or not"""
    binary_score: bool = Field(
        description="If the answer string actually answers the question then 'True' otherwise 'False'."
    )


structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n Give a binary score 'True' or 'False'. 'True' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: \n\n {question} \n\n Answer: {answer}")
    ]
)

answer_grader_chain:RunnableSequence = answer_prompt | structured_llm_answer_grader