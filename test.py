from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load model + processor
processor = TrOCRProcessor.from_pretrained("./trocr-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./trocr-finetuned")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load handwriting image (replace this with your file)
image = Image.open("data/smoothed.png").convert("RGB")

# Process image and run through model
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
print("Input shape:", pixel_values.shape)
generated_ids = model.generate(pixel_values, max_new_tokens=1000)
print(generated_ids)
# Decode and print the result
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Extracted Text:")
print(text)