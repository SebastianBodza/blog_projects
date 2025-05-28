import os
import logging
import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import librosa 

from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.tts import punc_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s3_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz").to(device)
voice_encoder = VoiceEncoder().to(device)
en_tokenizer = EnTokenizer("/home/bodza/blog_projects/09_Chatboxtts/chatterbox/tokenizer.json")

def process_audio(audio_path):
    """Load and preprocess audio file"""
    wav, sr = librosa.load(audio_path, sr=S3_SR)
    wav = torch.from_numpy(wav).float().to(device)
    return wav, sr

def extract_features(batch):
    """Extract features for a batch of samples"""
    features = {
        "speech_tokens": [],
        "speaker_embed": [],
        "text_tokens": [],
        "text": []
    }
    
    for audio_path, text in zip(batch["audio_path"], batch["text"]):
        try:
            # Process audio
            wav, sr = process_audio(audio_path)
            
            # Extract speech tokens
            speech_tokens, _ = s3_tokenizer.forward([wav], max_len=None)
            speech_tokens = speech_tokens[0].cpu().numpy()
            
            # Extract speaker embedding
            speaker_embed = voice_encoder.embeds_from_wavs(
                [wav.cpu().numpy()], 
                sample_rate=sr,
                as_spk=True
            )
            
            # Process text
            norm_text = punc_norm(text)
            text_tokens = en_tokenizer.encode(norm_text)
            
            features["speech_tokens"].append(speech_tokens)
            features["speaker_embed"].append(speaker_embed)
            features["text_tokens"].append(text_tokens)
            features["text"].append(norm_text)
            
        except Exception as e:
            logging.error(f"Error processing {audio_path}: {str(e)}")
    
    return features

def create_dataset(data_dir, output_path, batch_size=16):
    """Create preprocessed dataset"""
    # Load raw dataset
    from datasets import load_from_disk
    dataset = load_from_disk(data_dir)
    dataset = dataset["train"]
    
    # Process dataset in batches
    processed_data = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch in tqdm(dataloader, desc="Processing dataset"):
        features = extract_features(batch)
        processed_data.extend([
            {
                "speech_tokens": tokens,
                "speaker_embed": embed,
                "text_tokens": text_toks,
                "text": text
            }
            for tokens, embed, text_toks, text in zip(
                features["speech_tokens"],
                features["speaker_embed"],
                features["text_tokens"],
                features["text"]
            )
        ])
    
    # Create and save dataset
    hf_dataset = Dataset.from_list(processed_data)
    hf_dataset.save_to_disk(output_path)
    return hf_dataset

if __name__ == "__main__":
    data_dir = "/media/bodza/Audio_Dataset/MULTI_SPEAKER_PROCESSED/WHISPERX_DATASET"
    output_path = "/media/bodza/Audio_Dataset/chatterbox_tts_dataset"
    os.makedirs(output_path, exist_ok=True)
    create_dataset(data_dir, output_path)