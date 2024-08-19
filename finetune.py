import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from datasets import Dataset
from langchain_community.document_loaders import PyPDFLoader
from config import DATA_BASE, UPLOAD_FOLDER
def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()  # This loads the entire document
    text = ""
    for doc in documents:
        text += doc.page_content  # Assuming `page_content` contains the text
    return text

def load_and_prepare_data_from_pdfs(pdf_directory):
    texts = []
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            texts.append({"text": text})

    dataset = Dataset.from_dict({"text": texts})
    return dataset


def fine_tune_model(model_name, dataset, output_dir, num_train_epochs=3):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Setup QLoRA (Quantized LoRA)
    lora_config = LoraConfig(
        r=8,  # Low-rank adaptation parameter
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Which layers to apply Lora to
        lora_dropout=0.1,  # Dropout for Lora layers
        bias="none",  # Bias type
        task_type="CAUSAL_LM"  # Task type
    )
    
    model = get_peft_model(model, lora_config)

    # Tokenize the dataset
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
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
# Example usage
pdf_directory = DATA_BASE
dataset = load_and_prepare_data_from_pdfs(pdf_directory)
print(dataset)

# Example usage:
fine_tune_model(
    model_name="gpt2",
    dataset=dataset,
    output_dir="./fine_tuned_model",
    num_train_epochs=3
)
