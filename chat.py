from flask import jsonify
from modules.data_loader import DataLoader
from modules.vector_store_manager import VectorStoreManager
from modules.QAChain import QAChain
from modules.splitter import TextSplitter
from config import UPLOAD_FOLDER, DATA_BASE
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from datasets import Dataset
from langchain_community.document_loaders import PyPDFLoader
from config import DATA_BASE, UPLOAD_FOLDER
from finetune import load_and_prepare_data_from_pdfs

import mlflow
import mlflow.pytorch
class Process:
    def __init__(self):
        self.vector_store_manager = VectorStoreManager()
        self.model = None

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

    def fine_tune_model(self, model_name, output_dir, num_train_epochs):
        # Load the LLaMA 3 model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Prepare the model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Setup QLoRA (Quantized LoRA)
        lora_config = LoraConfig(
            r=8,  # Low-rank adaptation parameter
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # Which layers to apply Lora to
            lora_dropout=0.1,  # Dropout for Lora layers
            bias="none",  # Bias type
            task_type="CAUSAL_LM"  # Task type
        )
        
        self.model = get_peft_model(self.model, lora_config)

        # Tokenize the dataset
        dataset = load_and_prepare_data_from_pdfs(DATA_BASE)
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=8,
            num_train_epochs=num_train_epochs,
            logging_dir="./logs",
            logging_steps=500,
            save_steps=1000,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss"
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets,
            tokenizer=tokenizer
        )

        with mlflow.start_run():
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("output_dir", output_dir)
            mlflow.log_param("num_train_epochs", num_train_epochs)
            
            # Train the model
            trainer.train()

            # Save the model
            self.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Log the model to MLflow
            mlflow.pytorch.log_model(self.model, "model")

            mlflow.end_run()
