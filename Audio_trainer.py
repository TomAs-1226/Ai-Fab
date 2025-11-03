# audio_trainer.py
import os
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

# EXAMPLE TASK:
# Let's pretend it's keyword / command classification:
# each audio file -> label (class index).
# For ASR (speech-to-text), you'd instead load transcript
# and use CTC loss with a seq model head. :contentReference[oaicite:6]{index=6}

class AudioCommandDataset(Dataset):
    def __init__(self, manifest, sample_rate=16000, max_seconds=1.0):
        """
        manifest: list of dicts like
        {"wav_path": ".../yes_001.wav", "label": 0}
        We'll:
        - load waveform with torchaudio
        - resample to target SR
        - pad/trim to fixed length
        - return (audio_tensor, label)
        """
        self.items = manifest
        self.target_sr = sample_rate
        self.max_len = int(sample_rate * max_seconds)

        self.resampler = torchaudio.transforms.Resample(orig_freq=48000,
                                                        new_freq=sample_rate)

        # simple log-mel frontend (classic audio trick):
        # turning waveform -> time/freq representation model can learn
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=64,
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        wav, sr = torchaudio.load(entry["wav_path"])  # wav: [channels, time]
        if sr != self.target_sr:
            wav = self.resampler(wav)

        # mono
        wav = wav.mean(dim=0, keepdim=True)

        # pad / trim to self.max_len
        if wav.shape[-1] < self.max_len:
            pad_amt = self.max_len - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad_amt))
        else:
            wav = wav[..., : self.max_len]

        # mel features (log-mel spectrogram)
        mel = self.db(self.melspec(wav))  # [1, n_mels, time]

        label = torch.tensor(entry["label"], dtype=torch.long)
        return mel, label


class SmallAudioClassifier(nn.Module):
    """
    Tiny CNN on top of log-mel spectrograms.
    For real ASR you'd use Wav2Vec2 / Whisper-style encoders
    and maybe a CTC head. But this keeps it simple + fast. :contentReference[oaicite:7]{index=7}
    """
    def __init__(self, n_mels=64, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, mel):  # mel: [B, 1, n_mels, T]
        feats = self.net(mel)
        feats = feats.view(feats.size(0), -1)
        return self.fc(feats)


def train_audio(
    model,
    loader,
    num_epochs=10,
    lr=1e-3,
    grad_clip=1.0,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    ce = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for mel, label in loader:
            mel = mel.to(device)
            label = label.to(device)

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(mel)
                loss = ce(logits, label)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[audio] epoch {epoch+1} loss={avg_loss:.4f}")

    return model


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # You build `manifest` yourself from your dataset.
    manifest = [
        {"wav_path": r"C:\data\yes_001.wav", "label": 0},
        {"wav_path": r"C:\data\no_002.wav", "label": 1},
        # ...
    ]
    ds = AudioCommandDataset(manifest)
    dl = DataLoader(ds, batch_size=16, shuffle=True, pin_memory=True)

    clf = SmallAudioClassifier(n_mels=64, n_classes=2)
    train_audio(clf, dl)
