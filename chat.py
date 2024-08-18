from flask import jsonify
from modules.data_loader import DataLoader
from modules.vector_store_manager import VectorStoreManager
from modules.QAChain import QAChain
from modules.splitter import TextSplitter
from config import UPLOAD_FOLDER, DATA_BASE

class Process:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()

    def save(self):
        loader = DataLoader(loader_type="pdf", directory=UPLOAD_FOLDER)
        documents = loader.load_data()

        if not documents:
            print(f"No documents were loaded from {UPLOAD_FOLDER}.")
            return "No documents found or failed to load documents."

        print(f"Loaded {len(documents)} documents.")

        # Split documents into chunks
        text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(documents)

        # Create embeddings and store in vector store
        self.vector_store_manager = VectorStoreManager(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        self.vector_store_manager.store_embeddings([text.page_content for text in texts])

        if self.vector_store_manager.vector_store is None:
            return "Failed to create vector store. Please check the logs."

        self.vector_store_manager.save_local(DATA_BASE)

    def chat(self, user_input):
        # Load the vector store if not already loaded
        if self.vector_store_manager.vector_store is None:
            self.vector_store_manager.load_local(DATA_BASE)

        # Retrieve and run QA chain
        retriever = self.vector_store_manager.get_retriever()
        relevant_chunks = retriever.get_relevant_documents(user_input, top_k=5)
        qa_chain = QAChain(model_name="llama3", retriever=retriever)
        response, source_docs = qa_chain.run(query=user_input, context_chunks=relevant_chunks)

        return jsonify({
            "response": response, 
            "source_documents": [doc.page_content for doc in source_docs]
        })
