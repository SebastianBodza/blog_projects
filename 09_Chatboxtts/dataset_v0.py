import os
import logging
import numpy as np
import torch
from datasets import load_from_disk
import librosa 

from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.tts import punc_norm
from chatterbox.models.t3.modules.t3_config import T3Config 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s3_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz").to(device)
voice_encoder = VoiceEncoder().to(device)
en_tokenizer = EnTokenizer("/home/bodza/blog_projects/09_Chatboxtts/chatterbox/tokenizer.json")

def process_audio(audio_data):
    """Process audio from HF dataset format"""
    wav = audio_data["array"]
    sr = audio_data["sampling_rate"]
    
    if sr != S3_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=S3_SR)
    
    wav = torch.from_numpy(wav).float().to(device)
    return wav, S3_SR

def extract_features(sample):
    try:
        wav, sr = process_audio(sample["audio"])
        
        with torch.no_grad():
            speech_tokens, speech_token_lens = s3_tokenizer.forward([wav], max_len=None)
            speech_tokens = speech_tokens[0].cpu().numpy()
            speech_token_lens = speech_token_lens[0].cpu().item()
        
        with torch.no_grad():
            speaker_embed = voice_encoder.embeds_from_wavs(
                [wav.cpu().numpy()], 
                sample_rate=sr,
                as_spk=True
            )
        
        norm_text = punc_norm(sample["text"])
        text_tokens = en_tokenizer.encode(norm_text)

        hp_t3 = T3Config()
        prompt_len = hp_t3.speech_cond_prompt_len 

        cond_prompt_speech_tokens_np = speech_tokens[:prompt_len]

        return {
            "speech_tokens": speech_tokens,
            "speech_token_lens": speech_token_lens,
            "speaker_embed": speaker_embed,
            "text_tokens": text_tokens,
            "text_tokens_len": len(text_tokens),
            "text": norm_text,
            "cond_prompt_speech_tokens": cond_prompt_speech_tokens_np, 
            "cond_prompt_speech_tokens_len": len(cond_prompt_speech_tokens_np) 
        }
        
    except Exception as e:
        logging.error(f"Error processing sample: {str(e)}")
        return None

def create_dataset(data_dir, output_path, num_proc=4):
    """Create preprocessed dataset using map function"""
    logging.basicConfig(level=logging.INFO)
    
    dataset = load_from_disk(data_dir)
    dataset = dataset["train"]

    # limit for testing to 1000 samples
    # TODO: Removeme! 
    dataset = dataset.select(range(1000))

    # saving it to disk
    dataset.save_to_disk("./chatterbox_tts_dataset_raw")
    
    print(f"Processing {len(dataset)} samples...")
    
    processed_dataset = dataset.map(
        extract_features,
        num_proc=1, 
        desc="Processing dataset",
        remove_columns=dataset.column_names, 
    )
    
    processed_dataset = processed_dataset.filter(lambda x: x is not None)
    
    print(f"Successfully processed {len(processed_dataset)} samples")
    
    processed_dataset.save_to_disk(output_path)
    
    speech_lengths = [len(sample["speech_tokens"]) for sample in processed_dataset.select(range(min(100, len(processed_dataset))))]
    text_lengths = [sample["text_tokens_len"] for sample in processed_dataset.select(range(min(100, len(processed_dataset))))]
    
    print(f"Dataset saved to {output_path}")
    print(f"Total samples: {len(processed_dataset)}")
    print(f"Speech token lengths (first 100): min={min(speech_lengths)}, max={max(speech_lengths)}, mean={np.mean(speech_lengths):.1f}")
    print(f"Text token lengths (first 100): min={min(text_lengths)}, max={max(text_lengths)}, mean={np.mean(text_lengths):.1f}")
    
    return processed_dataset

if __name__ == "__main__":
    data_dir = "/media/bodza/Audio_Dataset/MULTI_SPEAKER_PROCESSED/WHISPERX_DATASET"
    output_path = "/media/bodza/Audio_Dataset/chatterbox_tts_dataset"
    os.makedirs(output_path, exist_ok=True)
    create_dataset(data_dir, output_path)
