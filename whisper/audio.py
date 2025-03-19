import torch
import numpy as np
import librosa

def load_audio(file_path, sr=16000):
    """Load an audio file and return a torch tensor."""
    audio, _ = librosa.load(file_path, sr=sr)
    return torch.from_numpy(audio).float()

def pad_or_trim(array, length):
    """Pad or trim an array to a specified length."""
    if len(array) > length:
        return array[:length]
    return np.pad(array, (0, length - len(array)))

def log_mel_spectrogram(audio, n_mels=80, n_fft=400, hop_length=160, sr=16000):
    """Convert audio waveform to log mel spectrogram."""
    audio = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0.0,
        fmax=sr/2.0
    )
    
    # Convert to log scale
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    
    # Normalize
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return torch.from_numpy(log_spec).float()

def prepare_audio_input(file_path, n_mels=80, sr=16000, max_length=30):
    """Load, process, and prepare audio for the Whisper model."""
    # Load audio
    audio = load_audio(file_path, sr=sr)
    
    # Pad or trim to max_length seconds
    max_samples = sr * max_length
    audio = pad_or_trim(audio.numpy(), max_samples)
    audio = torch.from_numpy(audio).float()
    
    # Extract log mel spectrogram
    mel = log_mel_spectrogram(audio, n_mels=n_mels)
    
    # Add batch dimension
    return mel.unsqueeze(0)
