import os
import torch
import soundfile as sf  # [HIGHLIGHT: NEW - to load audio from disk]
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

class AudioDataset(Dataset):
    def __init__(self, dataset, processor):
        """
        dataset: A Hugging Face dataset object with 'path' (string) and 'sentence' columns.
        processor: The Wav2Vec2Processor for feature extraction + tokenization.
        """
        self.dataset = dataset
        self.processor = processor

    def __getitem__(self, idx):
        # [HIGHLIGHT: CHANGED - no more sample["path"]["array"]]
        sample = self.dataset[idx]
        audio_path = sample["path"]         # a string path to the .mp3/.wav
        text = sample["sentence"]           # transcription string

        # 1. Check if file exists. If not, skip by returning None.
        if not os.path.exists(audio_path):
            return None

        # 2. Load the raw audio data using soundfile.
        audio_array, sr = sf.read(audio_path)

        # (Optional) If sr != 16000 and you want to resample, you'd do it here.
        # For now, we assume sr is already 16000 or close enough.

        # 3. Process / tokenize audio with Wav2Vec2Processor
        inputs = self.processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )

        # 4. Process the transcription into labels
        with self.processor.as_target_processor():
            labels = self.processor(text, return_tensors="pt").input_ids

        # 5. Format the item (removing extra batch dimension)
        item = {
            "input_values": inputs["input_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        return item

    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    """
    Collate function to merge variable-length audio sequences.
    We also drop any 'None' items from missing files.
    """
    # [HIGHLIGHT: CHANGED - filter out None items]
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # Edge case: if we skip a whole batch because all are missing
        return {
            "input_values": torch.empty(0),
            "attention_mask": torch.empty(0),
            "labels": torch.empty(0),
        }

    input_values = [b["input_values"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    # 1. Pad input_values and attention_mask
    input_values_padded = torch.nn.utils.rnn.pad_sequence(
        input_values, batch_first=True
    )
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {
        "input_values": input_values_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded
    }

def train_model(model, processor, train_dataset, epochs=1, batch_size=4):
    """
    Simplified training loop for Wav2Vec2-based CTC on audio data.
    This will skip missing audio files on the fly.
    """
    audio_dataset = AudioDataset(train_dataset, processor)
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
        for batch in loader:
            # If the entire batch is None or empty, skip
            if batch["input_values"].shape[0] == 0:
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
        
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model
