from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
from PIL import Image
import torch
import pandas as pd

# Load processor + model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# Load CSV
df = pd.read_csv("dataset.csv")
dataset = Dataset.from_pandas(df)

# Preprocessing function
def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
    labels = processor.tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    ).input_ids[0]
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore pad tokens
    return {"pixel_values": pixel_values, "labels": labels}

# Apply preprocessing
dataset = dataset.map(preprocess)

# Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./trocr-finetuned",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=100,
    logging_steps=50,
    predict_with_generate=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train
trainer.train()
trainer.save_model("./trocr-finetuned")
processor.save_pretrained("./trocr-finetuned")