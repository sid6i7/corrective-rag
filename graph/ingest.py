from typing import List, Tuple
import concurrent.futures
import json

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from utils import logger
from graph.chains.keyword_extractor import keyword_extractor_chain, DocumentKeywords


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
    
    def add_additional_metadata(self, documents:List[Document]) -> Tuple[Document, List[str]]:
        """Given a set of documents, generate relevant metadata for them."""
        logger.info("Generating metadata for documents")
        # Keyword extraction
        def get_keywords(document: Document) -> Document:
            try:
                document_keywords: DocumentKeywords = keyword_extractor_chain.invoke(
                    {"document": document.page_content}
                )
                document.metadata["keywords"] = json.dumps(document_keywords.keywords)
            except Exception as e:
                logger.error(f"Could not extract keywords for the document: {document.page_content}")
            return document
        documents_with_keywords = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(get_keywords, document) for document in documents]
            for future in futures:
                documents_with_keywords.append(future.result())
        return documents_with_keywords
        

    def load_documents(self, urls: List[str]) -> List[Document]:
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
    
    def add_documents_from_urls(self, urls: List[str]) -> VectorStoreRetriever | None:
        """
        Method to add new documents to an existing vector store
        """
        logger.info("Trying to add new documents to the vector store")
        try:
            documents:List[Document] = self.load_documents(urls)
            documents = self.add_additional_metadata(documents)
            self.vector_store.add_documents(documents)
            logger.info("Documents added succesfully")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
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