# Training Script
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.amp import GradScaler, autocast

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Using device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

caption_file = "C:/Users/ADARSH S/OneDrive/Desktop/GenAI Project/dataset/captions.txt"
image_folder = "C:/Users/ADARSH S/OneDrive/Desktop/GenAI Project/dataset/Images"

df = pd.read_csv(caption_file, sep=',', names=['image', 'caption'])
df['image'] = df['image'].apply(lambda x: x.split('#')[0])
df = df.groupby('image').first().reset_index()
df = df[df['image'].apply(lambda x: os.path.exists(os.path.join(image_folder, x)))]
print("Total usable images:", len(df))

class FlickrDataset(Dataset):
    def __init__(self, dataframe, image_dir, feature_extractor, tokenizer, max_length=16):
        self.df = dataframe
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        caption = row['caption']

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values[0]

        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0]
        }

batch_size = 16 
dataset = FlickrDataset(df, image_folder, feature_extractor, tokenizer)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scaler = GradScaler(device='cuda')

epochs = 10

for epoch in range(1,epochs):
    model.train()
    print(f"\nEpoch {epoch + 1}")
    total_loss = 0

    for batch in tqdm(train_loader):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            outputs = model(pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Average loss: {avg_loss:.4f}")

    # Save after every epoch
    output_dir = f"fine-tuned-captioning-epoch{epoch+1:02d}"
    model.save_pretrained(output_dir)
    feature_extractor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")
    
model.save_pretrained("vit-gpt2-captioning-model")
feature_extractor.save_pretrained("vit-gpt2-captioning-model")
tokenizer.save_pretrained("vit-gpt2-captioning-model")



