import os
# Define the GPU you want to run here with the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import base64
import logging
import re
import numpy as np
import librosa
import torch
import jiwer
import math
from typing import List
from fastapi import FastAPI, HTTPException
from models import TranscriptionRequest, TranscriptionResponse, DecodingRequest, DecodingResponse, CombinedRequest, CombinedResponse

from whisperx.asr import WhisperModel
from faster_whisper import BatchedInferencePipeline

from xcodec2.modeling_xcodec2 import XCodec2Model

from whisperx.audio import SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speech-service")

DEVICE = "cuda"
COMPUTE_TYPE = "int8"
DEFAULT_WHISPER_MODEL = "medium" 
XCODEC_MODEL_PATH = "HKUST-Audio/xcodec2"  

logger.info(f"Loading Whisper model '{DEFAULT_WHISPER_MODEL}' with batched inference pipeline...")
try:
    whisper_model = WhisperModel(
        DEFAULT_WHISPER_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    batched_pipeline = BatchedInferencePipeline(whisper_model)
    logger.info("Whisper model and batched inference pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

logger.info(f"Loading XCodec2 model from '{XCODEC_MODEL_PATH}' ...")
try:
    codec_model = XCodec2Model.from_pretrained(XCODEC_MODEL_PATH)
    codec_model.to(DEVICE)
    logger.info("XCodec2 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load XCodec2 model: {e}")
    raise


app = FastAPI(title="Speech Processing Service")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    start_time = time.time()
    try:
        audio_bytes = base64.b64decode(request.audio_data)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        
        if request.sample_rate != SAMPLE_RATE:
            audio_np = librosa.resample(audio_np, orig_sr=request.sample_rate, target_sr=SAMPLE_RATE)
        
        segments, info = batched_pipeline.transcribe(
            audio_np,
            language=request.language
        )
        text = " ".join(segment.text for segment in segments)
        processing_time = time.time() - start_time
        segments_out = [segment.__dict__ for segment in segments]
        return TranscriptionResponse(
            text=text,
            segments=segments_out,
            language=info.language,
            processing_time=processing_time,
        )
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_speech_ids(speech_tokens_str: str) -> List[int]:
    pattern = re.compile(r'<\|s_(\d+)\|>')
    matches = pattern.findall(speech_tokens_str)
    return [int(match) for match in matches]
        
@app.post("/decode", response_model=DecodingResponse)
async def decode_speech_tokens(request: DecodingRequest):
    start_time = time.time()
    try:

        speech_ids = extract_speech_ids(request.speech_tokens_str)

        if not speech_ids:
            raise HTTPException(status_code=400, detail="No valid speech tokens found.")
        
        with torch.no_grad():
            speech_tokens = torch.tensor(speech_ids).to(DEVICE).unsqueeze(0).unsqueeze(0)
            gen_wav = codec_model.decode_code(speech_tokens)
            audio_array = gen_wav[0, 0, :].cpu().numpy()
        
        audio_bytes = audio_array.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        processing_time = time.time() - start_time

        return DecodingResponse(
            audio_data=audio_b64,
            sample_rate=16000, 
            processing_time=processing_time,
        )
    except Exception as e:
        logger.error(f"Error during decoding: {e}")
        logger.error(f"Speech tokens: {request.speech_tokens_str}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/end_to_end", response_model=CombinedResponse)
async def end_to_end_processing(request: CombinedRequest):
    """Decode speech tokens and then transcribe the resulting audio"""
    start_time = time.time()
    
    decode_request = DecodingRequest(speech_tokens_str=request.speech_tokens_str)
    try:
        decoded = await decode_speech_tokens(decode_request)
    except HTTPException as e:
        raise e
    
    transcribe_request = TranscriptionRequest(
        audio_data=decoded.audio_data,
        sample_rate=decoded.sample_rate
    )
    try:
        transcribed = await transcribe_audio(transcribe_request)
    except HTTPException as e:
        raise e
    
    wer = None
    wer_reward = None
    if request.expected_text:
        normalized_answer = request.expected_text.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
        normalized_transcribed = transcribed.text.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "")
        
        wer = jiwer.wer(normalized_answer, normalized_transcribed)
        wer_reward = math.exp(-wer)

    hangup_reward = detect_hangups(transcribed.text)

    reward = None
    if wer_reward is not None:
        reward = (wer_reward + hangup_reward) / 2
    else:
        reward = hangup_reward
        
    processing_time = time.time() - start_time

    return CombinedResponse(
        transcribed_text=transcribed.text,
        wer=wer,
        wer_reward=wer_reward,
        hangup_reward=hangup_reward,
        reward=reward,
        audio_data=decoded.audio_data, 
        processing_time=processing_time
    )


# Reward function that detects hangups where a long word is generated like Ööööööööööööööööööööööööööööööööö... or Doooooooooooooooooooooooo
def detect_hangups(text: str, threshold=6, min_reward=0.0, max_reward=1.0) -> float:
    """
    Reward function that detects hangups where a long word is generated like Ööööööö...
    
    Args:
        text: Input string to check for repetitions
        threshold: Maximum allowed consecutive repetitions (default: 6)
        min_reward: Reward value for sequences exceeding threshold (default: 0.0)
        max_reward: Reward value for sequences below threshold (default: 1.0)
    
    Returns:
        float: max_reward for good sequences, min_reward for hangups
    """
    pattern = r'(.)\1{' + str(threshold-1) + r',}' 
    
    if re.search(pattern, text):
        print(f"Found excessive repetition. Applied min_reward: {min_reward}")
        return min_reward
    return max_reward



@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "whisper_model": DEFAULT_WHISPER_MODEL,
        "xcodec_model": XCODEC_MODEL_PATH,
        "compute_type": COMPUTE_TYPE,
    }


# run with fastapi run main.py --port 8080