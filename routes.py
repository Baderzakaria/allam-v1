from flask import Flask, request, jsonify
from modules.data_loader import DataLoader
from modules.embedder import TextEmbedder
from modules.vector_store_manager import VectorStoreManager
from modules.QAChain import QAChain
from modules.splitter import TextSplitter
from chat import process_chat
import os
from config import UPLOAD_FOLDER , ALLOWED_EXTENSIONS , DATA_BASE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"message": "No file part or selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], str(file.filename))
    file.save(filepath)
    return jsonify({"message": "File uploaded successfully", "filepath": filepath}), 200

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("message", "")
    if not user_input:
        return jsonify({"message": "Empty message received"}), 400
    # # Load documents
    # loader = DataLoader(loader_type='pdf', directory=app.config['UPLOAD_FOLDER'])
    # documents = loader.load_data()
    # if not documents:
    #     return jsonify({"message": "No documents found"}), 400

    # # Split documents into chunks
    # text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
    # texts = text_splitter.split_text(documents)

    # # Create embeddings and store in vector store
    # embedder = TextEmbedder()
    # vector_store_manager = VectorStoreManager(embedder)
    # print("vector")
    # vector_store_manager.store_embeddings([text.page_content for text in texts])
    # vector_store_manager.save_local(DATA_BASE)

    # # Retrieve and run QA chain
    # retriever = vector_store_manager.get_retriever()
    # qa_chain = QAChain(model_name="llama3", retriever=retriever)
    # response, source_docs = qa_chain.run(query=user_input, context_chunks=texts)
    response = process_chat(user_input, UPLOAD_FOLDER)

    return response
