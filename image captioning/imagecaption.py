

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)


device = "cpu"

model.to(device)

def generate_caption(image_path):
    # Open image
    image = Image.open(image_path).convert('RGB')

    # Preprocess
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate
    outputs = model.generate(**inputs, max_new_tokens=20)

    # Decode
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    return caption