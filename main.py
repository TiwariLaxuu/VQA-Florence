# Install required libraries
# !pip install datasets evaluate rouge_score

# Import necessary modules
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
import torch
from torchvision import transforms
import os

# comment this (RECOMMENDED) if you want to log the run to Weights & Biases
os.environ["WANDB_MODE"] = "offline" #

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Florence-2 model
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True).to(device)

# Freeze vision tower
for param in model.vision_tower.parameters():
    param.requires_grad = False

# Load processor
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

# Load dataset
ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")['raw']
dataset = ds.train_test_split(test_size=0.001)  ### using 1% of total data for DEMO only
dataset = dataset['test'].train_test_split(test_size=0.1) ### splitting 1% data into 90% train and 10% test
train_dataset, val_dataset = dataset['train'], dataset['test']

image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])

def collate_fn(batch):
    questions = [f"{x['question']}" for x in batch]
    # images = [x["image"].convert("RGB") if x["image"].mode != "RGB" else x["image"] for x in batch]
    #augument images 
    images = [image_transforms(x["image"].convert("RGB")) if x["image"].mode != "RGB" else x["image"] for x in batch]
    answers = [x["answer"] for x in batch]

    inputs = processor(text=questions, images=images, return_tensors="pt", padding=True)
    labels = processor.tokenizer(answers, return_tensors="pt", padding=True).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

import evaluate
import numpy as np

bleu, meteor, rouge = map(evaluate.load, ["bleu", "meteor", "rouge"])

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = processor.tokenizer.batch_decode(np.argmax(preds[0], axis=-1), skip_special_tokens=True)
    labels = processor.tokenizer.batch_decode(np.where(labels != -100, labels, processor.tokenizer.pad_token_id), skip_special_tokens=True)
    preds, labels = [p.strip() for p in preds], [l.strip() for l in labels]

    return {
        "bleu": bleu.compute(predictions=preds, references=[[l] for l in labels])["bleu"],
        "meteor": meteor.compute(predictions=preds, references=labels)["meteor"],
        "rougeL": rouge.compute(predictions=preds, references=labels)["rougeL"],
    }
    
training_args = TrainingArguments(
    output_dir="./Florence-2-vqa",
    per_device_train_batch_size=5, #### adjust as per GPU memory, 3 for Colab's T4
    num_train_epochs=3000,            #### adjust as per resources
    learning_rate=7.8e-6,
    weight_decay=0.1,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
    save_total_limit=3,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics # if you want to evaluate with NLP or other metrics (optional)
)
trainer.train()

model.save_pretrained("./Florence-2-vqa/final")

# Use your model repo name instead of /Florence-2-vqa-demo
model_hf = AutoModelForCausalLM.from_pretrained("./Florence-2-vqa/final", trust_remote_code=True).to(device)

