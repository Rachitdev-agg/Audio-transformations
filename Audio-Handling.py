from pathlib import Path
import torch
import torchaudio
from torchaudio import transforms
import matplotlib.pyplot as plt

torchaudio.set_audio_backend("soundfile")

current_dir = Path.cwd()
# print(type(current_dir))
audio_files = []


for item in current_dir.iterdir():
    # print(str(item))
    if str(item).endswith(".wav"):
        audio_files.append(item)

# for i, j in enumerate(audio_files):
#     print(i, j)

def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return sig, sr

def spectrogram(sig, sr, n_mels = 64, n_fft = 1024, hop_len = None):
    top_db = 80

    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec

sig, sr = open(audio_files[0])
spec = spectrogram(sig, sr)
print(spec.shape) 

plt.figure(figsize=(10, 4))
plt.imshow(spec.squeeze(), origin="lower", aspect="auto", cmap="magma")
plt.title("Mel Spectogram")
plt.xlabel("time")
plt.ylabel("frequency")
plt.colorbar(format="%+2.0f db")
plt.tight_layout()
plt.show()