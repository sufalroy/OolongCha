import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np
import sounddevice as sd
from threading import Event, Lock
from collections import deque
import queue

from .ai import STT, LLM, TTS, AudioConfig, ModelConfig, LLMConfig, TTSConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OolongCha:
    def __init__(self):
        self.input_device = None
        self.output_device = None
        self.input_rate = 16000
        self.output_rate = 16000
        
        self._setup_audio_devices()
        self._initialize_models()
        
        self.session_id = self.stt.create_session()
        self.is_running = False
        self.is_speaking = False
        self.stop_event = Event()
        self.audio_lock = Lock()
        self.audio_queue = queue.Queue(maxsize=10)
        self.buffer_size = 1024
        
    def _setup_audio_devices(self):
        devices = sd.query_devices()
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0 and self.input_device is None:
                if self._test_device_rate(i, 16000, is_input=True):
                    self.input_device = i
                    self.input_rate = 16000
                    logger.info(f"Selected input device {i}: {device['name']}")
                    break
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0 and self.output_device is None:
                if self._test_device_rate(i, 16000, is_input=False):
                    self.output_device = i
                    self.output_rate = 16000
                    logger.info(f"Selected output device {i}: {device['name']}")
                    break
        
        if self.input_device is None or self.output_device is None:
            raise RuntimeError("No suitable audio devices found")

    def _test_device_rate(self, device_id: int, rate: int, is_input: bool) -> bool:
        try:
            if is_input:
                sd.check_input_settings(device=device_id, samplerate=rate, channels=1)
            else:
                sd.check_output_settings(device=device_id, samplerate=rate, channels=1)
            return True
        except:
            return False

    def _initialize_models(self):
        audio_config = AudioConfig(rate=16000, chunk_size=1024)
        model_config = ModelConfig()
        llm_config = LLMConfig()
        tts_config = TTSConfig(output_sample_rate=16000)
        
        self.stt = STT(audio_config, model_config)
        self.llm = LLM(llm_config)
        self.tts = TTS(tts_config)

    def _audio_callback(self, indata, frames, time, status):
        if status:
            if status.input_overflow:
                logger.debug("Audio input overflow - skipping frame")
                return
            logger.warning(f"Audio callback status: {status}")
        
        if not self.is_speaking and self.is_running:
            try:
                audio_data = (indata[:, 0] * 32767).astype(np.int16)
                
                if not self.audio_queue.full():
                    self.audio_queue.put(audio_data.tobytes(), block=False)
                else:
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put(audio_data.tobytes(), block=False)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Audio callback error: {e}")

    def _audio_processor(self):
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                speech = self.stt.process_audio_chunk(self.session_id, audio_data)
                
                if speech and speech.text:
                    self._handle_user_input(speech.text)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio processor error: {e}")

    def _handle_user_input(self, text: str):
        logger.info(f"User: {text}")
        
        if text.lower().strip() in ['exit', 'quit', 'bye', 'goodbye']:
            self.stop()
            return
            
        if text.lower().strip() in ['clear', 'reset']:
            self.llm.clear_memory()
            self._speak("Memory cleared. How can I help you?")
            return
        
        try:
            message = self.llm.generate(text)
            if message.response:
                logger.info(f"Assistant: {message.response}")
                self._speak(message.response)
            else:
                self._speak("I didn't understand that. Could you please repeat?")
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            self._speak("I'm having trouble processing that. Please try again.")

    def _speak(self, text: str):
        if not text or not text.strip():
            return
            
        try:
            with self.audio_lock:
                self.is_speaking = True
                audio_data = self.tts.generate(text)
                
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
                
                sd.play(
                    audio_array.reshape(-1, 1),
                    samplerate=self.output_rate,
                    device=self.output_device
                )
                sd.wait()
                
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
        finally:
            self.is_speaking = False

    def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.stop_event.clear()
        
        audio_thread = threading.Thread(target=self._audio_processor, daemon=True)
        audio_thread.start()
        
        try:
            print("🎙️  Voice Assistant Starting...")
            print("💬 Say 'exit' or 'quit' to stop")
            print("🧠 Say 'clear' or 'reset' to clear memory")
            print("🔊 Listening...")
            
            self._speak("Hi! I'm your AI assistant. How can I help you today?")
            
            with sd.InputStream(
                callback=self._audio_callback,
                device=self.input_device,
                channels=1,
                samplerate=self.input_rate,
                blocksize=self.buffer_size,
                dtype=np.float32
            ):
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
            print(f"❌ Audio Error: {e}")
        finally:
            self.cleanup()

    def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        self._speak("Goodbye! Have a great day!")
        time.sleep(2)
        
        print("👋 Voice Assistant Stopped")

    def cleanup(self):
        try:
            if self.session_id:
                self.stt.remove_session(self.session_id)
            
            self.stt.cleanup()
            self.tts.cleanup()
            sd.stop()
            
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    try:
        app = OolongCha()
        app.start()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()