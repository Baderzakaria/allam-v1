from langchain_community.document_loaders import PyPDFLoader
import os

class DataLoader:
    def __init__(self, loader_type: str, directory: str, glob: str = "*.pdf"):
        self.loader_type = loader_type
        self.directory = directory
        self.glob = glob

    def load_data(self):
        documents = []
        if self.loader_type == "pdf":
            for file in os.listdir(self.directory):
                print(file)
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(self.directory, file))
                    documents.extend(loader.load())
        else:
            raise ValueError("Unsupported loader type")
        
        return documents
