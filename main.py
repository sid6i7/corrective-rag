from typing import List
import argparse
import os

from dotenv import load_dotenv
from langchain.schema import Document

from utils import logger, read_urls_from_file
from constants import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIRECTORY
from graph import app
from graph.ingest import RAGVectorStore


load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run corrective RAG with a question or URL ingestion.")
    
    parser.add_argument('-q', '--question', type=str, help="Input question for the RAG system")
    parser.add_argument('--urls', type=str, help="Path to text file containing URLs for ingestion")
    parser.add_argument('--collection_name', type=str, help="Name of the vector store collection for ingestion")
    parser.add_argument('--persist_directory', type=str, help="Path to the directory where vector store will be stored")

    args = parser.parse_args()
    
    collection_name = args.collection_name or DEFAULT_COLLECTION_NAME
    persist_directory = args.persist_directory or DEFAULT_PERSIST_DIRECTORY

    if args.urls:
        urls = read_urls_from_file(args.urls)
        vector_store = RAGVectorStore(collection_name, persist_directory)
        if not os.path.exists(persist_directory):
            documents = vector_store.load_documents(urls)
            doc_splits: List[Document] = vector_store.split_documents(documents)
            vector_store.create_vectorstore(doc_splits)
        else:
            vector_store.add_documents(urls)
    
    if args.question:
        if not os.path.exists(persist_directory):
            raise ValueError("Specified vector store doesn't exist")
        retriever = RAGVectorStore.get_retriever(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        logger.info(f"Querying with question: {args.question}")
        result = app.invoke(input={"question": args.question, "retriever": retriever})
        logger.info(result['answer'])
    else:
        logger.info("No question provided, URL ingestion completed.")


if __name__ == "__main__":
    main()