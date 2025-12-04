# Install necessary libraries
!pip install transformers accelerate soundfile scipy SpeechRecognition ipywidgets datasets -q

import torch
from transformers import pipeline
import soundfile as sf
import numpy as np
import io
from IPython.display import Audio, display
import speech_recognition as sr
from datasets import load_dataset

# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Part 1: Voice Cloning Setup (using a pre-trained TTS model with speaker conditioning) ---

print("Loading SpeechT5 models...")

# Load the SpeechT5 Text-to-Speech pipeline
# This pipeline can take a speaker embedding to condition the output voice.
synthesizer = pipeline("text-to-speech", "microsoft/speecht5_tts", device=device)

# Load the speaker verification model to extract speaker embeddings
# This model will generate an embedding from an audio input, which represents the speaker's voice characteristics.
speaker_embedding_model = "speechbrain/spkrec-ecapa-voxceleb"
speaker_verification_pipe = pipeline("speaker-verification", model=speaker_embedding_model, device=device)

print("Models loaded successfully.")

# --- Part 2: Extract Speaker Embedding from an Audio File ---

# In a Colab environment, you can upload an audio file.
# For demonstration, we'll use a pre-existing audio from a dataset.
# In a real scenario, you'd upload your target voice audio (e.g., a short clip of someone speaking).
print("
--- Extracting Speaker Embedding ---")
print("Please provide an audio file from which to clone the voice.")
print("You can upload one in Colab or specify a path to an existing file.")
print("For this example, we'll use a sample audio from the 'librispeech_asr' dataset.")

try:
    # Load a small sample from LibriSpeech for demonstration
    # You would replace this with your uploaded audio file.
    dataset = load_dataset("librispeech_asr", split="train.clean.100", streaming=True)
    sample_audio = next(iter(dataset))["audio"]
    target_audio_path = "sample_target_voice.wav"
    sf.write(target_audio_path, sample_audio["array"], sample_audio["sampling_rate"])
    print(f"Using sample audio: {target_audio_path}")
    print("Listen to the target voice:")
    display(Audio(target_audio_path, rate=sample_audio["sampling_rate"]))

    # Extract the speaker embedding from the target audio
    # This will generate a numerical representation of the voice characteristics.
    print("Extracting speaker embedding...")
    speaker_embedding_raw = speaker_verification_pipe(target_audio_path, return_embeddings=True)
    # The output from the speaker_verification_pipe is a list of dicts, so we extract the 'embeddings' tensor.
    # The embedding might be a list of tensors or a single tensor depending on the pipe's exact output.
    # We need to ensure it's a single tensor of shape [1, N] or [N] that can be used by SpeechT5.
    if isinstance(speaker_embedding_raw, list) and len(speaker_embedding_raw) > 0 and 'embeddings' in speaker_embedding_raw[0]:
        speaker_embedding = torch.tensor(speaker_embedding_raw[0]['embeddings']).to(device)
        # Ensure it has an extra dimension if needed by SpeechT5
        if speaker_embedding.dim() == 1:
            speaker_embedding = speaker_embedding.unsqueeze(0)
    else:
        raise ValueError("Could not extract embeddings in expected format.")

    print("Speaker embedding extracted.")

except Exception as e:
    print(f"Error loading or processing sample audio: {e}")
    print("Please ensure you have an audio file available and the dataset can be loaded.")
    speaker_embedding = None # In case of error, no embedding.

# --- Part 3: Generate New Speech with Cloned Voice ---

if speaker_embedding is not None:
    print("
--- Generating New Speech ---")
    text_to_clone = "Hello, this is a voice cloning demonstration in Google Colab. I can speak any text in this cloned voice."

    print(f"Text to synthesize: '{text_to_clone}'")
    print("Synthesizing speech...")

    # Generate speech using the synthesizer pipeline, conditioned by the extracted speaker embedding.
    # The 'speaker_embeddings' argument is crucial here for voice conditioning.
    # The pipeline returns a dictionary with 'audio' (numpy array) and 'sampling_rate'.
    speech_output = synthesizer(text_to_clone, forward_params={"speaker_embeddings": speaker_embedding})

    output_audio_path = "cloned_voice_output.wav"
    sf.write(output_audio_path, speech_output["audio"], speech_output["sampling_rate"])

    print(f"Generated speech saved to: {output_audio_path}")
    print("Listen to the cloned voice output:")
    display(Audio(output_audio_path, rate=speech_output["sampling_rate"]))
else:
    print("
Skipping speech generation as speaker embedding could not be extracted.")

print("
Voice cloning script finished.")
print("Remember to replace the sample audio with your desired target voice for actual cloning.")