from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from utils import logger


load_dotenv()


class RAGVectorStore:
    def __init__(self, collection_name: str, persist_directory: str):
        """
        Initialize the class with the URLs to fetch, the name of the vector store collection, and the directory for persistence.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=OpenAIEmbeddings(),
            )
        self.docs: List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )

    def load_documents(self, urls: List[str]):
        """
        Load web documents from the provided URLs.
        """
        try:
            logger.info("Loading documents from the web.")
            web_responses: List[List[Document]] = [WebBaseLoader(url).load() for url in urls]
            
            # Flatten the list of lists into a single list of documents
            documents = []
            for response in web_responses:
                documents.extend(response)
            logger.info(f"Loaded {len(documents)} documents.")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def split_documents(self, documents: List[Document]):
        """
        Split the loaded documents into smaller chunks.
        """
        if not documents:
            logger.error("No documents given. Cannot split documents.")
            raise ValueError("No documents given. Cannot split documents.")

        logger.info("Splitting documents into chunks.")
        doc_splits = self.text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(doc_splits)} chunks.")
        return doc_splits

    def create_vectorstore(self, doc_splits: List[Document]):
        """
        Create the Chroma vector store from the split documents and embeddings.
        """
        logger.info("Creating Chroma vector store.")
        try:
            vector_store = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding=OpenAIEmbeddings(),
            )
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
    
    def add_documents(self, urls: List[str]) -> VectorStoreRetriever | None:
        """
        Static method to add a new document to an existing vector store
        """
        logger.info("Trying to add a new document to the vector store")
        try:
            self.vector_store.add_documents(self.load_documents(urls))
            logger.info("Document added succesfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise