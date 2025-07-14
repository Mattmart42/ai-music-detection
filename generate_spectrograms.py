import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

def save_mel_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

for label in ['real', 'ai']:
    input_dir = f'data/{label}'
    output_dir = f'spectrograms/{label}'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            audio_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.wav', '.png'))
            save_mel_spectrogram(audio_path, output_path)
