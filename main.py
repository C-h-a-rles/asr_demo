from pathlib import Path
import os
import threading
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import numpy as np
import sounddevice as sd
import onnxruntime as ort

# =========================
# Windows 下尽量提前把 CUDA DLL 路径加入搜索路径
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

# 关键：先预加载 CUDA / cuDNN / MSVC DLL，再 import sherpa_onnx
ort.preload_dlls(cuda=True, cudnn=True, msvc=True)

import sherpa_onnx

# =========================
# 基础配置
# =========================
SAMPLE_RATE = 16000

# 改成你的麦克风索引；如果想用系统默认，改成 None
DEVICE_INDEX = 1

# 线程数
NUM_THREADS = 2

# 临时字幕刷新间隔（秒）
PARTIAL_UPDATE_INTERVAL = 0.12

# 每次读麦克风时长（秒）
READ_CHUNK_SECONDS = 0.10

# 最长保留的当前活动缓冲（秒）
MAX_ACTIVE_BUFFER_SECONDS = 12

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SENSEVOICE_DIR = MODEL_DIR / "sensevoice"

SENSEVOICE_MODEL = SENSEVOICE_DIR / "model.int8.onnx"
TOKENS = SENSEVOICE_DIR / "tokens.txt"
VAD_MODEL = MODEL_DIR / "silero_vad.onnx"


def check_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件: {path}")


def create_recognizer() -> sherpa_onnx.OfflineRecognizer:
    check_file(SENSEVOICE_MODEL)
    check_file(TOKENS)

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(SENSEVOICE_MODEL),
        tokens=str(TOKENS),
        num_threads=NUM_THREADS,
        use_itn=True,
        debug=False,
        provider="cuda",   # 关键：走 GPU
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


def get_input_devices():
    devices = sd.query_devices()
    lines = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            lines.append(f"[{i}] {d['name']}")
    return "\n".join(lines)


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


class ASRThread(threading.Thread):
    """
    模拟流式滚动字幕：
    - VAD 检测到开始说话后
    - 周期性对当前活动 buffer 重新解码，刷新 partial
    - 检测到停顿后，将 final 固化到历史字幕
    """
    def __init__(self, ui_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.audio_queue = queue.Queue()
        self.recorder = None

    def emit(self, kind, value):
        self.ui_queue.put((kind, value))

    def run(self):
        try:
            recognizer = create_recognizer()
            vad, window_size = create_vad()

            self.recorder = RecorderThread(self.audio_queue, self.stop_event)
            self.recorder.start()

            self.emit("status", f"识别中 | GPU | 输入设备: {get_current_device_name()}")

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

                # 按 VAD 窗口送入
                while offset + window_size < len(buffer):
                    vad.accept_waveform(buffer[offset: offset + window_size])

                    if not started and vad.is_speech_detected():
                        started = True
                        last_partial_time = time.time()

                    offset += window_size

                # 没开始说话时，只保留少量尾部，防止缓存无限长
                if not started:
                    keep = 10 * window_size
                    if len(buffer) > keep:
                        drop = len(buffer) - keep
                        buffer = buffer[-keep:]
                        offset = max(0, offset - drop)
                    continue

                # 说话过程中，周期性刷新临时字幕
                now = time.time()
                if last_partial_time is not None and (now - last_partial_time) >= PARTIAL_UPDATE_INTERVAL:
                    active_buffer = buffer[-max_active_samples:] if len(buffer) > max_active_samples else buffer
                    partial_text = self.decode_buffer(recognizer, active_buffer)
                    self.emit("partial", partial_text)
                    last_partial_time = now

                # 一旦 VAD 判断某段结束，则固化结果
                while not vad.empty():
                    segment = vad.front.samples
                    vad.pop()

                    final_text = self.decode_segment(recognizer, segment)
                    if final_text:
                        self.emit("final", final_text)

                    self.emit("partial", "")

                    buffer = np.array([], dtype=np.float32)
                    offset = 0
                    started = False
                    last_partial_time = None

            self.emit("status", "已停止")

        except Exception as e:
            self.emit("status", f"错误: {e}")

    @staticmethod
    def decode_buffer(recognizer, audio_buffer: np.ndarray) -> str:
        if len(audio_buffer) == 0:
            return ""

        t0 = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio_buffer)
        recognizer.decode_stream(stream)
        dt = (time.perf_counter() - t0) * 1000

        text = stream.result.text.strip()
        print(f"[partial decode] {dt:.1f} ms | {text}")
        return text

    @staticmethod
    def decode_segment(recognizer, segment: np.ndarray) -> str:
        if len(segment) == 0:
            return ""

        t0 = time.perf_counter()
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, segment)
        recognizer.decode_stream(stream)
        dt = (time.perf_counter() - t0) * 1000

        text = stream.result.text.strip()
        print(f"[final decode] {dt:.1f} ms | {text}")
        return text


class ASRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("本地滚动字幕式语音识别 Demo（GPU）")
        self.root.geometry("1020x760")

        self.ui_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.asr_thread = None

        self.history = []

        self.status_var = tk.StringVar(value="未启动")
        self.partial_var = tk.StringVar(value="")

        self.build_ui()
        self.root.after(50, self.poll_ui_queue)

    def build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=10, pady=10)

        tk.Button(top_frame, text="开始识别", width=12, command=self.start_recognition).pack(side="left", padx=5)
        tk.Button(top_frame, text="停止识别", width=12, command=self.stop_recognition).pack(side="left", padx=5)
        tk.Button(top_frame, text="清空文本", width=12, command=self.clear_text).pack(side="left", padx=5)
        tk.Button(top_frame, text="保存为 TXT", width=12, command=self.save_text).pack(side="left", padx=5)
        tk.Button(top_frame, text="复制全文", width=12, command=self.copy_text).pack(side="left", padx=5)

        status_frame = tk.Frame(self.root)
        status_frame.pack(fill="x", padx=10)

        tk.Label(status_frame, text="状态：", font=("Microsoft YaHei", 11)).pack(side="left")
        tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 11),
            fg="blue"
        ).pack(side="left")

        info_frame = tk.LabelFrame(self.root, text="输入设备信息", padx=8, pady=8)
        info_frame.pack(fill="x", padx=10, pady=10)

        info_text = f"当前 DEVICE_INDEX = {DEVICE_INDEX}\n\n可用输入设备：\n{get_input_devices()}"
        tk.Label(
            info_frame,
            text=info_text,
            justify="left",
            anchor="w",
            font=("Consolas", 10)
        ).pack(fill="x")

        history_frame = tk.LabelFrame(self.root, text="已确认字幕", padx=8, pady=8)
        history_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.history_text = ScrolledText(
            history_frame,
            wrap="word",
            font=("Microsoft YaHei", 12)
        )
        self.history_text.pack(fill="both", expand=True)

        partial_frame = tk.LabelFrame(self.root, text="当前滚动字幕", padx=8, pady=8)
        partial_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.partial_label = tk.Label(
            partial_frame,
            textvariable=self.partial_var,
            justify="left",
            anchor="w",
            wraplength=960,
            font=("Microsoft YaHei", 21, "bold"),
            fg="#b71c1c",
            bg="#fff8e1",
            padx=12,
            pady=12
        )
        self.partial_label.pack(fill="x")

    def start_recognition(self):
        if self.asr_thread is not None and self.asr_thread.is_alive():
            messagebox.showinfo("提示", "识别已经在运行中。")
            return

        self.stop_event = threading.Event()
        self.asr_thread = ASRThread(self.ui_queue, self.stop_event)
        self.asr_thread.start()
        self.status_var.set("启动中...")

    def stop_recognition(self):
        if self.asr_thread is None or not self.asr_thread.is_alive():
            self.status_var.set("未启动")
            return

        self.stop_event.set()
        self.status_var.set("停止中...")

    def clear_text(self):
        self.history.clear()
        self.history_text.delete("1.0", tk.END)
        self.partial_var.set("")

    def save_text(self):
        content = self.get_all_text().strip()
        if not content:
            messagebox.showwarning("提示", "当前没有可保存的文本。")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存识别结果",
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt")]
        )
        if not file_path:
            return

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        messagebox.showinfo("成功", f"已保存到：\n{file_path}")

    def copy_text(self):
        content = self.get_all_text().strip()
        if not content:
            messagebox.showwarning("提示", "当前没有可复制的文本。")
            return

        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        messagebox.showinfo("成功", "全文已复制到剪贴板。")

    def get_all_text(self):
        history = self.history_text.get("1.0", tk.END).strip()
        partial = self.partial_var.get().strip()
        if history and partial:
            return history + "\n" + partial
        return history or partial

    def poll_ui_queue(self):
        while not self.ui_queue.empty():
            kind, value = self.ui_queue.get_nowait()

            if kind == "status":
                self.status_var.set(value)
            elif kind == "partial":
                self.partial_var.set(value)
            elif kind == "final":
                self.append_final_text(value)

        self.root.after(50, self.poll_ui_queue)

    def append_final_text(self, text: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self.history.append(line)
        self.history_text.insert(tk.END, line + "\n")
        self.history_text.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = ASRApp(root)
    root.mainloop()