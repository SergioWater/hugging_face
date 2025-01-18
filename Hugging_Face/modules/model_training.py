"""
Contains the AudioDataset for training and the train_model() function
which orchestrates batching, training, and optional checkpoint saving.
"""

import os
import re
import torch
import soundfile as sf
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

class AudioDataset(Dataset):
    """
    A PyTorch Dataset that:
    1) Iterates over a Hugging Face Dataset (train_dataset).
    2) Loads audio from disk (path provided in each row).
    3) Cleans and tokenizes text via the Wav2Vec2 processor.

    'root_data_dir' is a string path to the *actual* data folder, e.g.:
    ".../Hugging_Face/data"
    We'll append "validated_clips" + filename to find the .wav.
    """

    def __init__(self, dataset, processor, root_data_dir):
        self.dataset = dataset
        self.processor = processor
        self.root_data_dir = Path(root_data_dir)  # Store as Path object
        self.skipped_count = 0

    def clean_text(self, text):
        """
        Converts text to lower, strips whitespace, and removes
        non-alphanumeric characters.
        """
        text = text.lower().strip()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text

    def __getitem__(self, idx):
        """
        Returns the processed sample at index idx, or None if the file
        doesn't exist or the audio is too short.
        """
        sample = self.dataset[idx]

        # We assume "path" is something like "clip_001.wav"
        audio_filename = sample["path"]
        text = sample["sentence"]

        # Build the FULL path with root_data_dir + validated_clips + filename
        full_audio_path = self.root_data_dir / "validated_clips" / audio_filename

        # If the file doesn't exist, skip
        if not full_audio_path.exists():
            self.skipped_count += 1
            return None

        # Load audio data
        audio_array, sr = sf.read(str(full_audio_path))
        if len(audio_array) < 160:
            self.skipped_count += 1
            return None

        # Clean text
        text = self.clean_text(text)
        if not text:
            self.skipped_count += 1
            return None

        # Process audio into input_values + attention_mask
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            text=text,
            return_tensors="pt",
            padding=True,
        )

        # Convert text to labels using the same processor
        with self.processor.as_target_processor():
            labels = self.processor(text, return_tensors="pt").input_ids


        return {
            "input_values":    inputs["input_values"].squeeze(0),
            "attention_mask":  inputs["attention_mask"].squeeze(0),
            "labels":          inputs["labels"].squeeze(0),
        }
    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    """
    A collation function that:
    1) Removes None items (skipped audio).
    2) Pads input_values, attention_mask, and labels to create uniform tensors.
    """
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

def train_model(
    model,
    processor,
    train_dataset,
    root_data_dir,
    epochs=1,
    batch_size=4,
    learning_rate=1e-5,
    save_checkpoint_dir=None,
):
    """
    Trains the Wav2Vec2 model for a given number of epochs on the train_dataset.
    This version ALWAYS performs a pre-check to see how many items would be skipped
    due to missing/invalid files (no option to skip this check).

    'root_data_dir': where the "validated_clips" folder is located.
                     Passed to AudioDataset so we can build full paths.
    """

    # Instantiate the dataset with the correct path to data
    audio_dataset = AudioDataset(train_dataset, processor, root_data_dir)
    print(f"Total items in dataset: {len(audio_dataset)}")

    # Always do the pre-check
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
        collate_fn=collate_fn,
        num_workers=0
    )

    optim = AdamW(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            # If an entire batch is empty, skip
            if batch["input_values"].shape[0] == 0:
                continue

            optim.zero_grad()

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss}")

        # Save checkpoint each epoch if requested
        if save_checkpoint_dir:
            epoch_dir = os.path.join(save_checkpoint_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)
            print(f"Checkpoint saved at: {epoch_dir}")

    return model