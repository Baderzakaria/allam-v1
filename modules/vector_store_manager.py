from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os
from config import UPLOAD_FOLDER , ALLOWED_EXTENSIONS , DATA_BASE


class VectorStoreManager:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
        self.vector_store = None
        self.logger = logging.getLogger(__name__)

    def store_embeddings(self, texts):
        try:
            self.logger.info("Storing embeddings...")
            # Store embeddings in the vector store using HuggingFaceEmbeddings
            self.vector_store = FAISS.from_texts(texts, embedding=self.embedder)
            if self.vector_store is not None:
                self.logger.info("Successfully stored embeddings in vector store.")
            else:
                self.logger.error("Failed to create vector store.")
        except Exception as e:
            self.logger.error(f"Failed to store embeddings: {e}")
            raise

    def answer_question(self, query, top_k=3):
        if self.vector_store is None:
            self.logger.error("Vector store is not initialized. Cannot answer the question.")
            raise ValueError("Vector store is not initialized. Please load or create the vector store first.")
        
        try:
            # Perform similarity search in the vector store
            results = self.vector_store.similarity_search(query, k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Failed to answer question: {e}")
            raise

    def save_local(self, path=DATA_BASE):
        try:
            if self.vector_store is None:
                self.logger.error("Vector store is not initialized. Cannot save.")
                raise ValueError("Vector store is not initialized.")

            # Create the directory if it doesn't exist
            if not os.path.exists(path):
                os.makedirs(path)
                self.logger.info(f"Directory {path} created for saving vector store.")

            # Save the vector store locally
            self.vector_store.save_local(path)
            self.logger.info(f"Vector store saved locally at {path}.")
        except Exception as e:
            self.logger.error(f"Failed to save vector store locally: {e}")
            raise

    def load_local(self, path=DATA_BASE):
        try:
            if not os.path.exists(path):
                self.logger.error(f"The path {path} does not exist. Cannot load vector store.")
                raise FileNotFoundError(f"The path {path} does not exist.")

            # Load the vector store from the local path
            self.vector_store = FAISS.load_local(path, embeddings=self.embedder)
            self.logger.info(f"Vector store loaded from local path: {path}.")
        except Exception as e:
            self.logger.error(f"Failed to load vector store from local path: {e}")
            raise

    def get_retriever(self):
        """Method to return a callable for similarity search."""
        if self.vector_store is None:
            self.logger.error("Vector store is not initialized. Cannot get retriever.")
            raise ValueError("Vector store is not initialized. Please load or create the vector store first.")
        retreiver = self.vector_store.as_retriever()
        return retreiver