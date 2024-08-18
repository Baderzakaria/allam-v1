from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def embed_query(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()  # Return embedding as a numpy array

    def embed_documents(self, texts):
        all_embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            all_embeddings.append(embedding)
        return np.vstack(all_embeddings)  # Stack embeddings into a single numpy array
