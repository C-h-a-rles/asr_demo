from pathlib import Path
import os
import time
import queue
import threading
import webbrowser
import re

import numpy as np
import sounddevice as sd
import onnxruntime as ort

# =========================
# Windows: 预加载 CUDA / cuDNN / MSVC DLL
# =========================
if hasattr(os, "add_dll_directory"):
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
            except Exception:
                pass

ort.preload_dlls(cuda=True, cudnn=True, msvc=True)

import sherpa_onnx

# =========================
# 你主要改这里
# =========================
DEVICE_INDEX = 1   # 改成你的麦克风索引；如果想用系统默认，改成 None
TARGET_URL = "https://www.bilibili.com/video/BV1XTWnewEdT/?spm_id_from=333.337.search-card.all.click&vd_source=43b77f575c20188b0dc14f5d6cb895fe"   # 改成你想打开的网址
WAKE_WORD = "Man."

# 冷却时间：避免一句话里连续触发很多次
COOLDOWN_SECONDS = 8.0

# 部分刷新间隔
PARTIAL_UPDATE_INTERVAL = 0.15

# 音频参数
SAMPLE_RATE = 16000
NUM_THREADS = 2
READ_CHUNK_SECONDS = 0.10
MAX_ACTIVE_BUFFER_SECONDS = 8

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SENSEVOICE_DIR = MODEL_DIR / "sensevoice"

SENSEVOICE_MODEL = SENSEVOICE_DIR / "model.int8.onnx"
TOKENS = SENSEVOICE_DIR / "tokens.txt"
VAD_MODEL = MODEL_DIR / "silero_vad.onnx"


def check_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件: {path}")


def normalize_text(text: str) -> str:
    """去掉空白和常见标点，降低匹配脆弱性"""
    text = text.strip().lower()
    text = re.sub(r"[，。！？；：、“”‘’（）()【】\[\]<>《》,.!?;:\s]+", "", text)
    return text


NORMALIZED_WAKE = normalize_text(WAKE_WORD)


def wake_word_hit(text: str) -> bool:
    return NORMALIZED_WAKE in normalize_text(text)


def create_recognizer() -> sherpa_onnx.OfflineRecognizer:
    check_file(SENSEVOICE_MODEL)
    check_file(TOKENS)

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(SENSEVOICE_MODEL),
        tokens=str(TOKENS),
        num_threads=NUM_THREADS,
        use_itn=True,
        debug=False,
        provider="cuda",
    )
    return recognizer


def create_vad():
    check_file(VAD_MODEL)

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(VAD_MODEL)
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.10
    config.silero_vad.min_speech_duration = 0.20
    config.silero_vad.max_speech_duration = 8
    config.sample_rate = SAMPLE_RATE

    vad = sherpa_onnx.VoiceActivityDetector(
        config=config,
        buffer_size_in_seconds=100,
    )
    return vad, config.silero_vad.window_size


def get_current_device_name():
    try:
        devices = sd.query_devices()
        if DEVICE_INDEX is None:
            idx = sd.default.device[0]
            return devices[idx]["name"]
        return devices[DEVICE_INDEX]["name"]
    except Exception:
        return "未知设备"


class RecorderThread(threading.Thread):
    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event

    def run(self):
        samples_per_read = int(READ_CHUNK_SECONDS * SAMPLE_RATE)

        stream_kwargs = dict(
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
        )
        if DEVICE_INDEX is not None:
            stream_kwargs["device"] = DEVICE_INDEX

        with sd.InputStream(**stream_kwargs) as stream:
            while not self.stop_event.is_set():
                samples, _ = stream.read(samples_per_read)
                samples = samples.reshape(-1).astype(np.float32)
                self.audio_queue.put(np.copy(samples))


class WakeBrowserApp:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.recorder = None
        self.last_trigger_time = 0.0

    def trigger(self, source_text: str):
        now = time.time()
        if now - self.last_trigger_time < COOLDOWN_SECONDS:
            return

        self.last_trigger_time = now
        print(f"\n[触发成功] 检测到唤醒词：{WAKE_WORD}")
        print(f"[来源文本] {source_text}")
        print(f"[动作] 打开浏览器 -> {TARGET_URL}\n")
        webbrowser.open(TARGET_URL)

    @staticmethod
    def decode(recognizer, audio: np.ndarray, tag: str) -> str:
        if len(audio) == 0:
            return ""

        t0 = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        dt = (time.perf_counter() - t0) * 1000

        text = stream.result.text.strip()
        print(f"[{tag}] {dt:.1f} ms | {text}")
        return text

    def run(self):
        recognizer = create_recognizer()
        vad, window_size = create_vad()

        self.recorder = RecorderThread(self.audio_queue, self.stop_event)
        self.recorder.start()

        print("==========================================")
        print("本地语音唤醒浏览器（GPU）")
        print(f"当前设备: {get_current_device_name()}")
        print(f"唤醒词: {WAKE_WORD}")
        print(f"目标网址: {TARGET_URL}")
        print("说出唤醒词后，将自动打开浏览器。")
        print("按 Ctrl+C 退出。")
        print("==========================================\n")

        buffer = np.array([], dtype=np.float32)
        offset = 0
        started = False
        last_partial_time = None
        max_active_samples = int(MAX_ACTIVE_BUFFER_SECONDS * SAMPLE_RATE)

        while not self.stop_event.is_set():
            try:
                samples = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            buffer = np.concatenate([buffer, samples])

            while offset + window_size < len(buffer):
                vad.accept_waveform(buffer[offset: offset + window_size])

                if not started and vad.is_speech_detected():
                    started = True
                    last_partial_time = time.time()

                offset += window_size

            if not started:
                keep = 10 * window_size
                if len(buffer) > keep:
                    drop = len(buffer) - keep
                    buffer = buffer[-keep:]
                    offset = max(0, offset - drop)
                continue

            now = time.time()
            if last_partial_time is not None and (now - last_partial_time) >= PARTIAL_UPDATE_INTERVAL:
                active_buffer = buffer[-max_active_samples:] if len(buffer) > max_active_samples else buffer
                partial_text = self.decode(recognizer, active_buffer, "partial")

                if partial_text and wake_word_hit(partial_text):
                    self.trigger(partial_text)

                last_partial_time = now

            while not vad.empty():
                segment = vad.front.samples
                vad.pop()

                final_text = self.decode(recognizer, segment, "final")

                if final_text and wake_word_hit(final_text):
                    self.trigger(final_text)

                buffer = np.array([], dtype=np.float32)
                offset = 0
                started = False
                last_partial_time = None


if __name__ == "__main__":
    app = WakeBrowserApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n已退出。")
