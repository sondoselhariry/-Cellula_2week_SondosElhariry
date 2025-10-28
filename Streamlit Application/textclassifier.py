import os
import torch
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel

# === Save Function
def save_model_components(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, encoder, save_path: str = "final_model"):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)  # Saves the LoRA adapter
    tokenizer.save_pretrained(save_path)
    joblib.dump(encoder, os.path.join(save_path, "label_encoder.pkl"))
    print(f"\nLoRA adapter + tokenizer + encoder saved to '{save_path}'")

def load_model_components(save_path="D:/SONDOS/NLP Cellula/WEEK 2/toxic-classifier/final_model"):
    # Load model directly from your final_model folder
    model = AutoModelForSequenceClassification.from_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    
    # Load encoder
    encoder_path = os.path.join(save_path, "label_encoder.pkl")
    encoder = joblib.load(encoder_path)

    model.eval()  
    return model, tokenizer, encoder


# === Inference Function ===
import torch.nn.functional as F

def classify_text(text, model, tokenizer, encoder):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
    predicted_class_id = probs.argmax()
    label = encoder.inverse_transform([predicted_class_id])[0]
    return label, dict(zip(encoder.classes_, probs.round(3)))


