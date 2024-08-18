import streamlit as st
import requests
import os

# Flask server URL
FLASK_URL = "http://0.0.0.0:5001"

def upload_pdf(file):
    if file is None:
        st.warning("Please upload a PDF file.")
        return None

    # Send file to Flask backend
    files = {"file": file}
    print("Sending file to Flask server...")  # Debugging
    response = requests.post(f"{FLASK_URL}/", files=files)
    print(f"Response status code: {response.status_code}")  # Debugging

    if response.status_code == 200:
        st.success("File uploaded successfully!")
    else:
        st.error(f"Failed to upload file. Response: {response.json()}")
    
    return response

def ask_question(question):
    data = {"message": question}
    response = requests.post(f"{FLASK_URL}/chat", json=data)
    print(f"Chat request status code: {response.status_code}")  # Debugging

    if response.status_code == 200:
        return response.json().get("response", "")
    else:
        st.error("Failed to get a response. Please try again.")
        return None

st.title("Chat with Your PDF")
st.write("Upload a PDF file and ask questions about its content.")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    upload_response = upload_pdf(uploaded_file)

    if upload_response:
        st.write("Now, you can ask a question about the document.")
        question = st.text_input("Ask your question here:")
        
        if st.button("Ask"):
            if question.strip():
                response = ask_question(question)
                if response:
                    st.write("### Answer:")
                    st.write(response)
            else:
                st.warning("Please enter a valid question.")
