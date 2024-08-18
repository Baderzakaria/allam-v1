from flask import Flask, request, jsonify
from chat import Process
import os
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

process = Process()

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"message": "No file part or selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], str(file.filename))
    file.save(filepath)

    process.save()
    return jsonify({"message": "File uploaded and embedded successfully", "filepath": filepath}), 200

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("message", "")
    if not user_input:
        return jsonify({"message": "Empty message received"}), 400
    
    response = process.chat(user_input)
    return response

