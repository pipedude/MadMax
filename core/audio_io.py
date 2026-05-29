import pyaudio
import asyncio
import queue
import threading
import logging
import math
import struct
import time
from config import (
    AUDIO_RATE,
    AUDIO_CHANNELS,
    AUDIO_CHUNK,
    AUDIO_PLAYBACK_DIAGNOSTICS,
)
from core.errors import AudioInitializationError
from core.state import AgentMode, AgentRuntimeState


logger = logging.getLogger(__name__)
_PCM16_BYTES_PER_SAMPLE = 2
_PLAYBACK_STARVATION_WARNING_MS = 20.0

class AudioIO:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.p = pyaudio.PyAudio()
        self.running = True  # <--- NEW: Application lifetime flag
        
        # MEMORY GUARD: Ring buffers (maxsize=50). Store ~4.2 sec of audio.
        self.mic_queue = asyncio.Queue(maxsize=50) 
        self.ww_queue = queue.Queue(maxsize=50) # Queue for local Vosk
        self.speaker_queue = queue.Queue()
        
        # ROUTER: "wakeword" (Sleep) or "api" (Live conversation)
        self.state = AgentRuntimeState()
        
        self.mic_stream = None
        self.speaker_stream = None
        self.speaker_thread = None
        self._speaker_chunk_seq = 0
        self._speaker_last_chunk_at = None
        self._speaker_last_write_started_at = None
        self._response_audio_started = False

    @property
    def routing_mode(self) -> AgentMode:
        return self.state.mode

    @routing_mode.setter
    def routing_mode(self, mode: AgentMode) -> None:
        self.state.mode = AgentMode(mode)

    @property
    def is_receiving_response(self) -> bool:
        return self.state.is_receiving_response

    @is_receiving_response.setter
    def is_receiving_response(self, value: bool) -> None:
        self.state.is_receiving_response = value
        if not value:
            self._response_audio_started = False

    @property
    def is_playing_audio(self) -> bool:
        return self.state.is_playing_audio

    @is_playing_audio.setter
    def is_playing_audio(self, value: bool) -> None:
        self.state.is_playing_audio = value

    def set_mode(self, mode: AgentMode) -> None:
        self.state.mode = mode

    def _bytes_to_duration_ms(self, size_bytes: int) -> float:
        return (size_bytes / (AUDIO_RATE * AUDIO_CHANNELS * _PCM16_BYTES_PER_SAMPLE)) * 1000.0

    def enqueue_speaker_audio(self, data: bytes) -> None:
        self.speaker_queue.put(data)
        if not AUDIO_PLAYBACK_DIAGNOSTICS:
            return
        now = time.monotonic()
        self._speaker_chunk_seq += 1
        gap_ms = None if self._speaker_last_chunk_at is None else (now - self._speaker_last_chunk_at) * 1000.0
        self._speaker_last_chunk_at = now
        duration_ms = self._bytes_to_duration_ms(len(data))
        logger.info(
            "[PlaybackDiag] chunk_received seq=%s bytes=%s duration_ms=%.1f queue_depth=%s buffered_ms=%.1f gap_ms=%s",
            self._speaker_chunk_seq,
            len(data),
            duration_ms,
            self.speaker_queue.qsize(),
            self._bytes_to_duration_ms(sum(len(chunk) for chunk in list(self.speaker_queue.queue) if isinstance(chunk, bytes))),
            "n/a" if gap_ms is None else f"{gap_ms:.1f}",
        )

    def start(self):
        try:
            self.mic_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK,
                stream_callback=self._mic_callback
            )
            self.mic_stream.start_stream()

            self.speaker_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE,
                output=True,
                frames_per_buffer=AUDIO_CHUNK
            )
            
            self.speaker_thread = threading.Thread(target=self._speaker_worker, daemon=True)
            self.speaker_thread.start()
        except (OSError, RuntimeError, ValueError) as exc:
            raise AudioInitializationError(f"Failed to initialize audio: {exc}") from exc

    def _mic_callback(self, in_data, frame_count, time_info, status):
        try:
            # 1. Sleep mode: send audio to local Vosk detector
            if self.routing_mode == AgentMode.WAKEWORD:
                try:
                    self.ww_queue.put_nowait(in_data)
                except queue.Full:
                    try:
                        self.ww_queue.get_nowait()
                        self.ww_queue.put_nowait(in_data)
                    except queue.Empty:
                        pass

            # 2. Active mode: send audio to Gemini Live servers
            elif self.routing_mode == AgentMode.API:
                if not self.is_receiving_response and not self.is_playing_audio:
                    def _safe_put():
                        try:
                            self.mic_queue.put_nowait(in_data)
                        except asyncio.QueueFull:
                            try:
                                self.mic_queue.get_nowait()
                                self.mic_queue.put_nowait(in_data)
                            except asyncio.QueueEmpty:
                                return
                    self.loop.call_soon_threadsafe(_safe_put)
            
            return (None, pyaudio.paContinue)
        
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("Microphone dropout: %s", exc)
            return (None, pyaudio.paAbort)

    def _speaker_worker(self):
        while True:
            wait_started_at = time.monotonic()
            data = self.speaker_queue.get()
            wait_ms = (time.monotonic() - wait_started_at) * 1000.0
            if data is None: 
                break 
            if self.is_receiving_response and self._response_audio_started and wait_ms >= _PLAYBACK_STARVATION_WARNING_MS:
                logger.warning(
                    "Playback buffer starvation detected: wait_ms=%.1f queue_depth=%s receiving_response=%s",
                    wait_ms,
                    self.speaker_queue.qsize(),
                    self.is_receiving_response,
                )
            
            self.is_playing_audio = True
            self._speaker_last_write_started_at = time.monotonic()
            
            try:
                self.speaker_stream.write(data)
                self._response_audio_started = True
            except (OSError, RuntimeError, ValueError) as exc:
                logger.error("Speaker dropout: %s", exc)
                self.is_playing_audio = False
                break 

            if AUDIO_PLAYBACK_DIAGNOSTICS:
                write_elapsed_ms = (time.monotonic() - self._speaker_last_write_started_at) * 1000.0
                logger.info(
                    "[PlaybackDiag] chunk_written bytes=%s write_ms=%.1f queue_depth_after=%s buffered_ms_after=%.1f receiving_response=%s",
                    len(data),
                    write_elapsed_ms,
                    self.speaker_queue.qsize(),
                    self._bytes_to_duration_ms(sum(len(chunk) for chunk in list(self.speaker_queue.queue) if isinstance(chunk, bytes))),
                    self.is_receiving_response,
                )
            
            if self.speaker_queue.empty():
                self.is_playing_audio = False

    def play_beep(self):
        """Plays a pleasant wake-up tone (pure 440Hz sine wave)."""
        frames =[]
        for i in range(int(AUDIO_RATE * 0.2)):  # 0.2 seconds
            value = int(math.sin(2 * math.pi * 440 * i / AUDIO_RATE) * 10000)
            frames.append(struct.pack("<h", value))
        self.speaker_queue.put(b"".join(frames))

    def clear_wakeword_queue(self):
        while not self.ww_queue.empty():
            try:
                self.ww_queue.get_nowait()
            except queue.Empty:
                break

    def clear_speaker_queue(self):
        while not self.speaker_queue.empty():
            try:
                self.speaker_queue.get_nowait()
            except queue.Empty:
                break

    def clear_mic_queue(self):
        while not self.mic_queue.empty():
            try:
                self.mic_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def reset_state(self):
        """Instant state reset: drop flags and clear queues."""
        self.state.reset()
        self._speaker_chunk_seq = 0
        self._speaker_last_chunk_at = None
        self._speaker_last_write_started_at = None
        self._response_audio_started = False
        
        self.clear_speaker_queue()
        self.clear_mic_queue()

    def close(self):
        """Safe shutdown (even if threads already crashed)."""
        self.running = False  # <--- NEW: Tell background threads it's time to exit
        
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Error closing microphone stream.", exc_info=True)
            
        self.speaker_queue.put(None)
        
        if self.speaker_stream:
            try:
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
            except (OSError, RuntimeError, ValueError):
                logger.debug("Error closing output audio stream.", exc_info=True)
            
        try:
            self.p.terminate()
        except (OSError, RuntimeError, ValueError):
            logger.debug("Error terminating PyAudio.", exc_info=True)