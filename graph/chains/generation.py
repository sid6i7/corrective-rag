from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from langchain_openai import ChatOpenAI

from graph.constants import DEFAULT_COMPLETIONS_MODEL


llm = ChatOpenAI(model=DEFAULT_COMPLETIONS_MODEL, temperature=0)
prompt = hub.pull("rlm/rag-prompt")

generation_chain:RunnableSequence = prompt | llm | StrOutputParser()