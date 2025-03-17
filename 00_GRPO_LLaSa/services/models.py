from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class TranscriptionRequest(BaseModel):
    audio_data: str  
    sample_rate: int
    language: Optional[str] = None

class TranscriptionResponse(BaseModel):
    text: str
    segments: List[Dict[str, Any]]
    language: str
    processing_time: float

class DecodingRequest(BaseModel):
    speech_tokens_str: str  

class DecodingResponse(BaseModel):
    audio_data: str 
    sample_rate: int
    processing_time: float


class CombinedRequest(BaseModel):
    speech_tokens_str: str
    expected_text: Optional[str] = None

class CombinedResponse(BaseModel):
    transcribed_text: str
    wer: Optional[float] = None
    reward: Optional[float] = None
    audio_data: Optional[str] = None 
    processing_time: float
    wer_reward: Optional[float] = None
    hangup_reward: Optional[float] = None
