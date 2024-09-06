import logging
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGVectorStore:
    def __init__(self, urls: List[str], collection_name: str, persist_directory: str):
        """
        Initialize the class with the URLs to fetch, the name of the vector store collection, and the directory for persistence.
        """
        self.urls = urls
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.docs: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )

    def load_documents(self):
        """
        Load web documents from the provided URLs.
        """
        try:
            logger.info("Loading documents from the web.")
            web_responses: List[List[Document]] = [WebBaseLoader(url).load() for url in self.urls]
            
            # Flatten the list of lists into a single list of documents
            for response in web_responses:
                self.docs.extend(response)
                
            logger.info(f"Loaded {len(self.docs)} documents.")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def split_documents(self):
        """
        Split the loaded documents into smaller chunks.
        """
        if not self.docs:
            logger.error("No documents loaded. Cannot split documents.")
            raise ValueError("No documents to split. Make sure to load documents first.")

        logger.info("Splitting documents into chunks.")
        doc_splits = self.text_splitter.split_documents(self.docs)
        logger.info(f"Split documents into {len(doc_splits)} chunks.")
        return doc_splits

    def create_vectorstore(self, doc_splits: List[Document]):
        """
        Create the Chroma vector store from the split documents and embeddings.
        """
        logger.info("Creating Chroma vector store.")
        try:
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding=OpenAIEmbeddings(),
            )
            vectorstore.persist()  # Persist the store for later use
            logger.info("Vector store created and persisted.")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    @staticmethod
    def get_retriever(collection_name: str, persist_directory: str):
        """
        Static method to return a retriever for querying the vector store, 
        independent of the class initialization.
        """
        logger.info("Initializing retriever from the Chroma vector store.")
        try:
            retriever = Chroma(
                collection_name=collection_name,
                persist_directory=persist_directory,
                embedding_function=OpenAIEmbeddings(),
            ).as_retriever()
            logger.info("Retriever initialized successfully.")
            return retriever
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

"""Example usage:
if __name__ == "__main__":
    # urls = [
    #     "https://www.prompthub.us/blog/how-to-use-system-2-attention-prompting-to-improve-llm-accuracy",
    #     "https://www.prompthub.us/blog/enterprise-prompt-engineering-best-practices-tendencies-and-insights",
    #     "https://www.prompthub.us/blog/least-to-most-prompting-guide",
    #     "https://www.prompthub.us/blog/self-consistency-and-universal-self-consistency-prompting",
    #     "https://www.prompthub.us/blog/prompt-patterns-what-they-are-and-16-you-should-know",
    #     "https://www.prompthub.us/blog/program-of-thoughts-prompting-guide"
    # ]

    # vector_store = RAGVectorStore(
    #     urls=urls, 
    #     collection_name="rag-chroma", 
    #     persist_directory="./.chroma"
    # )

    # vector_store.load_documents()
    # doc_splits = vector_store.split_documents()
    # vector_store.create_vectorstore(doc_splits)

    # Access the retriever
    retriever = RAGVectorStore.get_retriever(
        collection_name="rag-chroma", 
        persist_directory="./.chroma"
    )
    # Now you can use the retriever for querying the vector store.
"""