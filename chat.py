from flask import jsonify
from modules.data_loader import DataLoader
from modules.vector_store_manager import VectorStoreManager
from modules.QAChain import QAChain
from modules.splitter import TextSplitter
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, DATA_BASE


def process_chat(user_input, upload_folder):
    # Load documents
    loader = DataLoader(loader_type="pdf", directory=upload_folder)
    documents = loader.load_data()

    if not documents:
        print(f"No documents were loaded from {upload_folder}.")
        return "No documents found or failed to load documents."

    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(documents)

    # Create embeddings and store in vector store
    vector_store_manager = VectorStoreManager(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    vector_store_manager.store_embeddings([text.page_content for text in texts])
    
    if vector_store_manager.vector_store is None:
        return "Failed to create vector store. Please check the logs."

    vector_store_manager.save_local(DATA_BASE)

    # Retrieve and run QA chain
    retriever = vector_store_manager.get_retriever()
    qa_chain = QAChain(model_name="llama3", retriever=retriever)
    response, source_docs = qa_chain.run(query=user_input, context_chunks=texts)

    return jsonify({
        "response": response, 
        "source_documents": [doc.page_content for doc in source_docs]
    })
