# [model_training.py]
import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

class AudioDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.skipped_count = 0

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        audio_path = sample["path"]
        text = sample["sentence"].lower().strip()

        full_audio_path = os.path.join(
            "/Users/water/Documents/Coding/Ai/hugging_face/Hugging_Face/data",
            "validated_clips",
            audio_path
        )

        # Skip if file doesn't exist
        if not os.path.exists(full_audio_path):
            self.skipped_count += 1
            return None

        audio_array, sr = sf.read(full_audio_path)
        if len(audio_array) < 160:
            self.skipped_count += 1
            return None

        if not text:
            self.skipped_count += 1
            return None

        # Make the processor return an attention_mask
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )

        # Now handle text => labels, no as_target_processor
        labels = self.processor(
            text=text,
            return_tensors="pt"
        ).input_ids

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    # Filter out any None items
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {
            "input_values": torch.empty(0),
            "attention_mask": torch.empty(0),
            "labels": torch.empty(0),
        }

    input_values = [b["input_values"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_values": input_values_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

def train_model(model, processor, train_dataset, epochs=1, batch_size=4):

    audio_dataset = AudioDataset(train_dataset, processor)
    print(f"Total items in dataset: {len(audio_dataset)}")

    # Quick pass to see how many items might be skipped
    skip_test_count = 0
    for i in range(len(audio_dataset)):
        item = audio_dataset[i]
        if item is None:
            skip_test_count += 1
    print(f"Would skip {skip_test_count} items out of {len(audio_dataset)}")

    loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    optim = AdamW(model.parameters(), lr=1e-4)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            if batch["input_values"].shape[0] == 0:
                # Empty batch => skip
                continue

            optim.zero_grad()

            input_values = batch["input_values"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optim.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss}")

    return model
