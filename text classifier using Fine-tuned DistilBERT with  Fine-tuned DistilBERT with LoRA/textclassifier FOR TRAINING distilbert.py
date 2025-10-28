

import pandas as pd

df = pd.read_csv("D:/SONDOS/NLP Cellula/WEEK 2/dataset2.csv")

#simple data cleaning
df['prompt'] = df['prompt'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
labels=df['label']
ytrue = encoder.fit_transform(labels)
print(ytrue)
# Check for class count
print(df['label'].value_counts())

# to find the number of unique class labels
print("Unique labels:", df['label'].nunique())

print(encoder.classes_)

import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

from sklearn.model_selection import train_test_split



train_texts, val_texts, train_labels, val_labels  = train_test_split(
   df['prompt'].tolist(),ytrue.tolist(),
    test_size=0.2,
    stratify=df['label'],            # ensures class balance in both sets
    random_state=42
)

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import torch

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(encoder.classes_))

# Increase rank for more capacity
lora_config = LoraConfig(
    r=16,  # Increased from 8
    lora_alpha=32,  # Increased from 16
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "ffn.lin1", "ffn.lin2"],
    task_type=TaskType.SEQ_CLS,
    lora_dropout=0.1,  # Slightly higher dropout
    bias="all"  # Add bias tuning
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",           # Evaluate after each epoch
    save_strategy="epoch",           # Save after each epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,                # Log every 50 steps
    report_to="none",
    load_best_model_at_end=True,     # Optional: load best model at the end
    metric_for_best_model="accuracy" # Optional: use accuracy to determine best model
)

from datasets import Dataset
train_dataset_hf = Dataset.from_dict({
    "text": train_texts,
    "label": train_labels
})

val_dataset_hf = Dataset.from_dict({
    "text": val_texts,
    "label": val_labels
})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

train_dataset_hf = train_dataset_hf.map(tokenize_function, batched=True)
val_dataset_hf = val_dataset_hf.map(tokenize_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
     }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=val_dataset_hf,
    compute_metrics=compute_metrics,  # Add this line
    data_collator=data_collator
)

train_result = trainer.train()

eval_results = trainer.evaluate()
print("\n=== Validation Results ===")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

predictions = trainer.predict(val_dataset_hf)
print(f"\nPrediction metrics: {predictions.metrics}")

import os
import torch
import pickle
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel, PeftConfig

trainer.model = trainer.model.merge_and_unload()

# === Save Function ===
def save_model_components(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, encoder, save_path: str = "final_model"):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)  # Only saves the LoRA adapter
    tokenizer.save_pretrained(save_path)
    joblib.dump(encoder, os.path.join(save_path, "label_encoder.pkl"))
    print(f"\nLoRA adapter + tokenizer + encoder saved to '{save_path}'")

def load_model_components(save_path):
    # Load base model and tokenizer from base checkpoint
    base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Load LoRA-adapted model weights
    model = PeftModel.from_pretrained(base_model, save_path)

    # Load encoder using joblib (fixes unpickling error)
    encoder_path = os.path.join(save_path, "label_encoder.pkl")
    encoder = joblib.load(encoder_path)

    return model, tokenizer, encoder

# === Inference Function ===
def classify_text(text, model, tokenizer, encoder):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return encoder.inverse_transform([predicted_class_id])[0]

save_model_components(trainer.model, tokenizer, encoder, save_path="final_model")