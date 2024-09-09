from typing import List

from langchain.schema import Document

from graph.re_patterns import EXTRA_WHITESPACE_PATTERN


def preprocess_keywords(keywords: List[str]) -> List[str]:
    processed_keywords = [keyword.replace("-", " ").lower() for keyword in keywords]
    return processed_keywords


def preprocess_documents(documents: List[Document]) -> List[Document]:
    def trim_extra_space(input_str):
        lines = input_str.split("\n")
        trimmed_lines = [EXTRA_WHITESPACE_PATTERN.sub(r"\1", line.strip()) for line in lines if line.strip()]
        return "\n".join(trimmed_lines)
    for document in documents:
        document.page_content = trim_extra_space(document.page_content)
    return documents
        