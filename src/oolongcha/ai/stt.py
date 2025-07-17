from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import time
from uuid import uuid4
import torch
from whisper_trt import load_trt_model, set_cache_dir
from whisper_trt.vad import load_vad
import logging
from collections import deque
from threading import Lock
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AudioConfig:
    rate: int = 16000
    min_silence_duration: float = 0.8
    max_silence_duration: float = 1.5
    speech_patience: float = 3.0
    min_speech_duration: float = 0.3
    noise_threshold: float = 0.15
    max_history_seconds: float = 20.0
    vad_window_size: int = 6
    chunk_size: int = 1536

@dataclass(frozen=True)
class ModelConfig:
    name: str = "small.en"
    cache_dir: Optional[str] = "models/whisper_trt"

@dataclass(slots=True)
class Speech:
    id: str
    text: str
    start_time: float
    end_time: float
    is_final: bool = True

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_final": self.is_final
        }

class AudioProcessor:
    __slots__ = ('config', 'buffer', 'vad_window', 'is_speaking', 'speech_start', 
                 'silence_start', 'lock')

    def __init__(self, config: AudioConfig):
        self.config = config
        self.buffer = deque(maxlen=int(config.max_history_seconds * config.rate))
        self.vad_window = deque(maxlen=config.vad_window_size)
        self.is_speaking = False
        self.speech_start = None
        self.silence_start = None
        self.lock = Lock()

    def process(self, audio_chunk: np.ndarray, vad_score: float) -> Optional[np.ndarray]:
        with self.lock:
            self.buffer.extend(audio_chunk)
            self.vad_window.append(vad_score)
            avg_vad = np.mean(self.vad_window)
            should_process = self._update_state(avg_vad)
            return self._get_and_clear_buffer() if should_process else None

    def _update_state(self, vad_score: float) -> bool:
        current_time = time.time()
        
        if vad_score > self.config.noise_threshold:
            if not self.is_speaking:
                self.speech_start = current_time
                self.is_speaking = True
                self.silence_start = None
            return False

        if not self.is_speaking:
            return False

        if self.silence_start is None:
            self.silence_start = current_time

        silence_duration = current_time - self.silence_start
        speech_duration = current_time - self.speech_start

        if (silence_duration >= self.config.min_silence_duration and 
            (silence_duration >= self.config.max_silence_duration or 
             speech_duration >= self.config.speech_patience)):
            self.is_speaking = False
            return True
        return False

    def _get_and_clear_buffer(self) -> np.ndarray:
        audio = np.array(self.buffer)
        self.buffer.clear()
        self.vad_window.clear()
        self.speech_start = None
        self.silence_start = None
        return audio

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()
            self.vad_window.clear()
            self.is_speaking = False
            self.speech_start = None
            self.silence_start = None

class STT:
    _IGNORED_PHRASES = {
        'blank_audio', 'background noise', 'pause', 'silence', 'inaudible',
        'speaking in foreign language', 'foreign language', 'coughing',
        'people chattering', 'muffled speaking', 'background conversations'
    }

    def __init__(self, audio_cfg: AudioConfig, model_cfg: ModelConfig):
        self.audio_cfg = audio_cfg
        self.model_cfg = model_cfg
        self.processors: Dict[str, AudioProcessor] = {}
        self._setup_models()

    def _setup_models(self) -> None:
        if self.model_cfg.cache_dir:
            set_cache_dir(self.model_cfg.cache_dir)

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True

        self.vad = load_vad()
        self.model = load_trt_model(self.model_cfg.name)
        self._warmup()

    def _warmup(self) -> None:
        dummy_audio = np.zeros(self.audio_cfg.chunk_size, dtype=np.float32)
        self.vad(dummy_audio, sr=self.audio_cfg.rate)
        self.model.transcribe(dummy_audio)

    def create_session(self) -> str:
        session_id = str(uuid4())
        self.processors[session_id] = AudioProcessor(self.audio_cfg)
        return session_id

    def remove_session(self, session_id: str) -> None:
        if session_id in self.processors:
            self.processors[session_id].clear()
            del self.processors[session_id]

    @staticmethod
    @lru_cache(maxsize=512)
    def _convert_audio(audio_data: bytes) -> np.ndarray:
        return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    def process_audio_chunk(self, session_id: str, audio_data: bytes) -> Optional[Speech]:
        if session_id not in self.processors:
            return None

        try:
            audio = self._convert_audio(audio_data)
            vad_score = float(self.vad(audio, sr=self.audio_cfg.rate).flatten()[0])
            
            processor = self.processors[session_id]
            speech_audio = processor.process(audio, vad_score)

            if (speech_audio is not None and 
                len(speech_audio) >= self.audio_cfg.rate * self.audio_cfg.min_speech_duration):
                return self._transcribe(speech_audio)
            return None

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None

    def _transcribe(self, audio: np.ndarray) -> Optional[Speech]:
        try:
            audio = self._denoise(audio)
            result = self.model.transcribe(audio)
            
            text = result.get('text', '').strip()
            if not text or self._is_ignored_phrase(text):
                return None

            current_time = time.time()
            return Speech(
                id=str(uuid4()),
                text=text,
                start_time=current_time - len(audio) / self.audio_cfg.rate,
                end_time=current_time,
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _is_ignored_phrase(self, text: str) -> bool:
        if ((text.startswith('[') and text.endswith(']')) or 
            (text.startswith('(') and text.endswith(')'))):
            inner_text = text[1:-1].strip().lower()
            return inner_text in self._IGNORED_PHRASES
        return False

    def _denoise(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        abs_audio = np.abs(audio)
        noise_floor = np.percentile(abs_audio, 15)
        mask = abs_audio > noise_floor
        
        denoised = np.zeros_like(audio)
        denoised[mask] = audio[mask] * (1.0 - noise_floor / abs_audio[mask])
        return denoised

    def cleanup(self) -> None:
        for session_id in list(self.processors.keys()):
            self.remove_session(session_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()