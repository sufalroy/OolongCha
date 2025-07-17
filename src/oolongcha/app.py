import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Optional
import numpy as np
import sounddevice as sd
from threading import Event, Thread

from ai import STT, LLM, TTS, AudioConfig, ModelConfig, LLMConfig, TTSConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OolongCha:
    def __init__(self):
        self.audio_config = AudioConfig()
        self.model_config = ModelConfig()
        self.llm_config = LLMConfig()
        self.tts_config = TTSConfig()
        
        self.stt = STT(self.audio_config, self.model_config)
        self.llm = LLM(self.llm_config)
        self.tts = TTS(self.tts_config)
        
        self.session_id = self.stt.create_session()
        self.is_running = False
        self.is_speaking = False
        self.audio_thread = None
        self.stop_event = Event()
        
        self.input_device = None
        self.output_device = None
        self._setup_audio_devices()

    def _setup_audio_devices(self):
        try:
            devices = sd.query_devices()
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0 and self.input_device is None:
                    self.input_device = i
                if device['max_output_channels'] > 0 and self.output_device is None:
                    self.output_device = i
                    
            logger.info(f"Audio devices - Input: {self.input_device}, Output: {self.output_device}")
            
        except Exception as e:
            logger.error(f"Audio device setup failed: {e}")
            raise

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        if not self.is_speaking and self.is_running:
            audio_data = (indata[:, 0] * 32767).astype(np.int16)
            speech = self.stt.process_audio_chunk(self.session_id, audio_data.tobytes())
            
            if speech and speech.text:
                self._handle_user_input(speech.text)

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
                logger.info(f"Sky: {message.response}")
                self._speak(message.response)
            else:
                self._speak("I'm sorry, I didn't understand that. Could you please repeat?")
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            self._speak("I'm having trouble processing that. Please try again.")

    def _speak(self, text: str):
        if not text or not text.strip():
            return
            
        try:
            self.is_speaking = True
            audio_data = self.tts.generate(text)
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767.0
            
            sd.play(
                audio_array.reshape(-1, 1),
                samplerate=self.tts_config.output_sample_rate,
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
        
        try:
            print("🎙️  Sky AI Assistant Starting...")
            print("💬 Say 'exit' or 'quit' to stop")
            print("🧠 Say 'clear' or 'reset' to clear memory")
            print("🔊 Listening...")
            
            self._speak("Hi! I'm Sky, your AI assistant. How can I help you today?")
            
            with sd.InputStream(
                callback=self._audio_callback,
                device=self.input_device,
                channels=1,
                samplerate=self.audio_config.rate,
                blocksize=self.audio_config.chunk_size,
                dtype=np.float32
            ):
                while self.is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
        finally:
            self.cleanup()

    def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        self.stop_event.set()
        
        self._speak("Goodbye! Have a great day!")
        time.sleep(2)
        
        print("👋 Sky AI Assistant Stopped")

    def cleanup(self):
        try:
            if self.session_id:
                self.stt.remove_session(self.session_id)
            
            self.stt.cleanup()
            self.tts.cleanup()
            
            sd.stop()
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def run_interactive(self):
        try:
            self.start()
        except KeyboardInterrupt:
            self.stop()

def main():
    try:
        app = OolongCha()
        app.run_interactive()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()