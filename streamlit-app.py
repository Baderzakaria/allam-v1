import streamlit as st
import requests

# Flask server URL
FLASK_URL = "http://0.0.0.0:5001"

def upload_pdf(file):
    if file is None:
        st.warning("Please upload a PDF file.")
        return None

    # Send file to Flask backend
    files = {"file": file}
    print("Sending file to Flask server...")  # Debugging
    response = requests.post(f"{FLASK_URL}/upload", files=files)
    print(f"Response status code: {response.status_code}")  # Debugging

    if response.status_code == 200:
        st.success("File uploaded successfully!")
        return True
    else:
        st.error(f"Failed to upload file. Response: {response.json()}")
        return False

def ask_question(question):
    data = {"message": question}
    response = requests.post(f"{FLASK_URL}/chat", json=data)
    print(f"Chat request status code: {response.status_code}")  # Debugging

    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        st.error("Failed to get a response. Please try again.")
        return None

def fine_tune_model(model_name, output_dir, num_train_epochs):
    data = {
        "model_name": model_name,
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs
    }
    response = requests.post(f"{FLASK_URL}/fine-tune", json=data)
    print(f"Fine-tune request status code: {response.status_code}")  # Debugging

    if response.status_code == 200:
        st.success("Model fine-tuned and saved successfully!")
    else:
        st.error("Failed to fine-tune the model. Please try again.")

# Streamlit UI
st.title("Chat with Your PDF")
st.write("Upload a PDF file and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Add an upload button
if st.button("Upload PDF"):
    if uploaded_file is not None:
        upload_success = upload_pdf(uploaded_file)
        if upload_success:
            st.write("Now, you can ask a question about the document.")
    else:
        st.warning("Please upload a file before clicking the upload button.")

question = st.text_input("Ask your question here:")

# Add a button to ask a question
if st.button("Ask"):
    response = ask_question(question)
    if response:
        st.write("### Answer:")
        st.write(response)

# Fine-tune button
if st.button("Fine-Tune Model"):
    model_name = "llama3"  # Replace with your model name
    output_dir = "./fine_tuned_model"
    num_train_epochs = 3
    fine_tune_model(model_name, output_dir, num_train_epochs)
