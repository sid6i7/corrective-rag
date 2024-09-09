import argparse
import os

from dotenv import load_dotenv

from utils import logger, read_urls_from_file
from constants import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIRECTORY
from graph import app
from graph.ingest import RAGVectorStore


load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Run corrective RAG with a question or URL ingestion.")
    
    parser.add_argument('-q', '--question', type=str, help="Input question for the RAG system")
    parser.add_argument('--urls', type=str, help="Path to text file containing URLs for ingestion")
    parser.add_argument(
        '--collection_name', type=str,
        help=f"Name of the vector store collection for ingestion. Defaults to {DEFAULT_COLLECTION_NAME}",
        default=DEFAULT_COLLECTION_NAME
    )
    parser.add_argument(
        '--persist_directory', type=str,
        help=f"Path to the directory where vector store will be stored. Defaults to {DEFAULT_PERSIST_DIRECTORY}",
        default=DEFAULT_PERSIST_DIRECTORY
    )

    args = parser.parse_args()
    
    collection_name = args.collection_name
    persist_directory = args.persist_directory

    if args.urls:
        logger.info("Attempting to add urls..")
        urls = read_urls_from_file(args.urls)
        logger.info(f"Found {len(urls)} from {args.urls}")
        vector_store = RAGVectorStore(collection_name, persist_directory)
        vector_store.add_documents_from_urls(urls)
    
    if args.question:
        if not os.path.exists(persist_directory):
            raise ValueError("Specified vector store doesn't exist")
        retriever = RAGVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        logger.info(f"Querying with question: {args.question}")
        result = app.invoke(input={"question": args.question, "retriever": retriever})
        logger.info(f"Answer: {result['answer']}")
    else:
        logger.info("No question provided, URL ingestion completed.")


if __name__ == "__main__":
    main()