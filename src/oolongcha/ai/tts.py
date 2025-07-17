import logging
from dataclasses import dataclass
import os
from typing import Optional, Tuple
import numpy as np
import librosa
from kokoro_onnx import Kokoro
from onnxruntime import InferenceSession
import onnxruntime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSError(Exception):
    pass

@dataclass(frozen=True)
class TTSConfig:
    model_path: str = "models/kokoro/kokoro-v1.0.fp16.onnx"
    voices_path: str = "models/kokoro/voices-v1.0.bin"
    voice: str = "af_sky"
    output_sample_rate: int = 16000
    speed: float = 1.0

class TTS:    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.model = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = os.cpu_count()
            session = InferenceSession(
                self.config.model_path, 
                providers=['CPUExecutionProvider'], 
                sess_options=sess_options
            )
            self.model = Kokoro.from_session(session, self.config.voices_path)
            logger.info("TTS model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            raise TTSError(f"Model initialization failed: {e}") from e

    @staticmethod
    def _resample(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        if original_sr == target_sr:
            return audio
        return librosa.resample(
            y=audio.astype(np.float32),
            orig_sr=original_sr,
            target_sr=target_sr
        )

    def _process_audio(self, samples: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        samples = np.clip(samples, -1, 1)
        
        if sample_rate != self.config.output_sample_rate:
            samples = self._resample(samples, sample_rate, self.config.output_sample_rate)
            sample_rate = self.config.output_sample_rate

        return samples, sample_rate

    def generate(self, text: str, voice: Optional[str] = None, language: str = "en-us") -> bytes:
        if not text or not text.strip():
            raise TTSError("Empty text provided")
            
        try:
            selected_voice = voice or self.config.voice
            
            samples, sample_rate = self.model.create(
                text=text.strip(),
                voice=selected_voice,
                speed=self.config.speed,
                lang=language
            )
            
            samples, sample_rate = self._process_audio(samples, sample_rate)           
            samples = (samples * 32767).astype(np.int16)
            return samples.tobytes()
                
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            raise TTSError(f"Audio generation failed: {e}") from e

    def cleanup(self) -> None:
        if self.model is not None:
            self.model = None
            logger.info("TTS model cleaned up")