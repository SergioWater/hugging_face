# [[ model_training.py ]]

import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

class AudioDataset(Dataset):
    def __init__(self, dataset, processor):
        print("========== [DEBUG] ENTER AudioDataset.__init__ ==========")
        self.dataset = dataset
        self.processor = processor
        # We'll track how many items we skip
        self.skipped_count = 0
        print("========== [DEBUG] EXIT AudioDataset.__init__ ==========")

    def __getitem__(self, idx):
        print(f"========== [DEBUG] ENTER AudioDataset.__getitem__ with idx={idx} ==========")
        sample = self.dataset[idx]
        audio_path = sample["path"]               # a string path to the audio file
        text = sample["sentence"]                 # transcript string

        # 1. Lowercase transcript to match typical Wav2Vec2 vocab
        text = text.lower()
        print(f"========== [DEBUG] after lowercasing, text={text} ==========")

        # 2. Check if file is on disk
        if not os.path.exists(audio_path):
            print(f"[WARN] Missing file at {audio_path}. Skipping.")
            self.skipped_count += 1
            return None

        # 3. Load audio data from disk
        audio_array, sr = sf.read(audio_path)
        print(f"========== [DEBUG] sf.read => sr={sr}, len(audio_array)={len(audio_array)} ==========")

        # Potential skip if too short
        if len(audio_array) < 160:  # e.g., <0.01s at 16kHz
            print(f"[WARN] Very short audio => skipping idx={idx}")
            self.skipped_count += 1
            return None

        # Potential skip if transcript is empty
        if not text.strip():
            print(f"[WARN] Empty transcript => skipping idx={idx}")
            self.skipped_count += 1
            return None

        # 4. Process / tokenize audio with Wav2Vec2Processor
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        print("========== [DEBUG] Audio processed by Wav2Vec2Processor ==========")

        # 5. Process the transcript into labels
        with self.processor.as_target_processor():
            labels = self.processor(text, return_tensors="pt").input_ids
        print("========== [DEBUG] Text processed into labels ==========")

        item = {
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        print(f"========== [DEBUG] EXIT AudioDataset.__getitem__ idx={idx} (return item) ==========")
        return item

    def __len__(self):
        length = len(self.dataset)
        print(f"========== [DEBUG] AudioDataset.__len__ => {length} ==========")
        return length

def collate_fn(batch):
    print("========== [DEBUG] ENTER collate_fn ==========")
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        print("[WARN] Entire batch was None or empty, returning empty dict.")
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

    print(f"========== [DEBUG] collate_fn returning batch size={len(batch)} ==========")
    print("========== [DEBUG] EXIT collate_fn ==========")
    return {
        "input_values": input_values_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

def train_model(model, processor, train_dataset, epochs=1, batch_size=4):
    print("========== [DEBUG] ENTER train_model ==========")
    print(f"========== [DEBUG] train_model called with epochs={epochs}, batch_size={batch_size} ==========")

    audio_dataset = AudioDataset(train_dataset, processor)
    loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    print("========== [DEBUG] DataLoader created ==========")

    optim = AdamW(model.parameters(), lr=1e-4)
    print("========== [DEBUG] AdamW optimizer created (lr=1e-4) ==========")
    model.train()
    print("========== [DEBUG] Model set to train() mode ==========")

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        print(f"========== [DEBUG] Starting epoch {epoch+1}/{epochs} ==========")

        for batch_idx, batch in enumerate(loader):
            print(f"========== [DEBUG] ENTER training loop batch_idx={batch_idx} ==========")
            if batch["input_values"].shape[0] == 0:
                print(f"[WARN] skipping empty batch batch_idx={batch_idx}")
                continue

            optim.zero_grad()

            input_values = batch["input_values"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            print(f"========== [DEBUG] input_values.shape={input_values.shape}, labels.shape={labels.shape} ==========")

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

            print(f"[DEBUG] Batch {batch_idx} => loss={loss.item():.4f}")
            print(f"========== [DEBUG] EXIT training loop batch_idx={batch_idx} ==========")

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Show how many items were actually skipped
    print(f"[DEBUG] Done training. Skipped items total: {audio_dataset.skipped_count}")
    print("========== [DEBUG] EXIT train_model ==========")
    return model
