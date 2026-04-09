from pathlib import Path
import os
import sys
import threading
import queue
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import webbrowser
import re

import numpy as np
import sounddevice as sd
import onnxruntime as ort


# =========================
# 路径与运行环境
# =========================
def get_app_dir() -> Path:
    # 打包后：exe 所在目录
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # 源码运行：当前 py 文件所在目录
    return Path(__file__).resolve().parent


def get_bundle_dir() -> Path:
    # onefile 模式下，PyInstaller 解包临时目录
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return get_app_dir()


APP_DIR = get_app_dir()
BUNDLE_DIR = get_bundle_dir()

# 模型优先从 exe 同级目录找，其次尝试 bundle 目录
MODEL_DIR_CANDIDATES = [
    APP_DIR / "models",
    BUNDLE_DIR / "models",
]

# GPU DLL 优先从 exe 同级目录找，其次尝试 bundle 目录
GPU_DLL_DIR_CANDIDATES = [
    APP_DIR / "gpu_dlls",
    BUNDLE_DIR / "gpu_dlls",
]


def first_existing_dir(candidates):
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


MODEL_DIR = first_existing_dir(MODEL_DIR_CANDIDATES) or (APP_DIR / "models")
GPU_DLL_DIR = first_existing_dir(GPU_DLL_DIR_CANDIDATES)

SENSEVOICE_DIR = MODEL_DIR / "sensevoice"
SENSEVOICE_MODEL = SENSEVOICE_DIR / "model.int8.onnx"
TOKENS = SENSEVOICE_DIR / "tokens.txt"
VAD_MODEL = MODEL_DIR / "silero_vad.onnx"


# =========================
# Windows: DLL 搜索路径与 ORT 预加载
# =========================
def setup_windows_dll_search():
    if not hasattr(os, "add_dll_directory"):
        return

    added = set()

    def add_dir(p: Path | None):
        if p and p.exists() and p.is_dir():
            sp = str(p.resolve())
            if sp not in added:
                try:
                    os.add_dll_directory(sp)
                    added.add(sp)
                except Exception:
                    pass

    # 1) 先加 exe 目录
    add_dir(APP_DIR)

    # 2) 再加 gpu_dlls 目录
    add_dir(GPU_DLL_DIR)

    # 3) 再尝试系统 CUDA_PATH/bin
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        add_dir(Path(cuda_path) / "bin")


def preload_ort_runtime():
    # 优先从 gpu_dlls 目录预加载；没有则走默认搜索策略
    try:
        if GPU_DLL_DIR and GPU_DLL_DIR.exists():
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory=str(GPU_DLL_DIR))
        else:
            ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    except Exception:
        # 不在这里抛，后面创建 recognizer 时再兜底
        pass


setup_windows_dll_search()
preload_ort_runtime()

import sherpa_onnx


# =========================
# 基础配置
# =========================
SAMPLE_RATE = 16000
DEVICE_INDEX = 1          # 改成你的麦克风索引；如果想用系统默认，改成 None
NUM_THREADS = 2

PARTIAL_UPDATE_INTERVAL = 0.12
READ_CHUNK_SECONDS = 0.10
MAX_ACTIVE_BUFFER_SECONDS = 12
COOLDOWN_SECONDS = 8.0    # 唤醒后冷却，防止重复打开网页


# =========================
# 通用工具
# =========================
def check_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件: {path}")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[，。！？；：、“”‘’（）()【】\[\]<>《》,.!?;:\s]+", "", text)
    return text


def wake_word_hit(text: str, wake_word: str) -> bool:
    return normalize_text(wake_word) in normalize_text(text)


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


# =========================
# 模型创建
# =========================
def create_recognizer():
    check_file(SENSEVOICE_MODEL)
    check_file(TOKENS)

    # 先试 GPU，再回退 CPU
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(SENSEVOICE_MODEL),
            tokens=str(TOKENS),
            num_threads=NUM_THREADS,
            use_itn=True,
            debug=False,
            provider="cuda",
        )
        return recognizer, "cuda"
    except Exception as e:
        print(f"CUDA init failed, fallback to CPU: {e}")
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(SENSEVOICE_MODEL),
            tokens=str(TOKENS),
            num_threads=NUM_THREADS,
            use_itn=True,
            debug=False,
            provider="cpu",
        )
        return recognizer, "cpu"


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


# =========================
# 录音线程
# =========================
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


# =========================
# ASR 线程
# =========================
class ASRThread(threading.Thread):
    def __init__(
        self,
        ui_queue: queue.Queue,
        stop_event: threading.Event,
        wake_enabled: bool,
        wake_word: str,
        target_url: str,
    ):
        super().__init__(daemon=True)
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.audio_queue = queue.Queue()
        self.recorder = None

        self.wake_enabled = wake_enabled
        self.wake_word = wake_word.strip()
        self.target_url = target_url.strip()
        self.last_trigger_time = 0.0

    def emit(self, kind, value):
        self.ui_queue.put((kind, value))

    def maybe_trigger_browser(self, source_text: str):
        if not self.wake_enabled or not self.wake_word or not self.target_url:
            return
        if not wake_word_hit(source_text, self.wake_word):
            return

        now = time.time()
        if now - self.last_trigger_time < COOLDOWN_SECONDS:
            return

        self.last_trigger_time = now
        self.emit("trigger", f"命中唤醒词：{self.wake_word}，正在打开：{self.target_url}")
        try:
            webbrowser.open(self.target_url)
        except Exception as e:
            self.emit("status", f"打开浏览器失败: {e}")

    def run(self):
        try:
            recognizer, provider_used = create_recognizer()
            vad, window_size = create_vad()

            self.recorder = RecorderThread(self.audio_queue, self.stop_event)
            self.recorder.start()

            mode_text = "启用" if self.wake_enabled else "关闭"
            provider_text = "GPU" if provider_used == "cuda" else "CPU"
            self.emit(
                "status",
                f"识别中 | {provider_text} | 输入设备: {get_current_device_name()} | 网页唤醒: {mode_text}"
            )

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
                    partial_text = self.decode_buffer(recognizer, active_buffer)
                    self.emit("partial", partial_text)
                    if partial_text:
                        self.maybe_trigger_browser(partial_text)
                    last_partial_time = now

                while not vad.empty():
                    segment = vad.front.samples
                    vad.pop()

                    final_text = self.decode_segment(recognizer, segment)
                    if final_text:
                        self.emit("final", final_text)
                        self.maybe_trigger_browser(final_text)

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


# =========================
# GUI
# =========================
class ASRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("本地滚动字幕式语音识别 Demo（打包兼容版）")
        self.root.geometry("1080x900")

        self.ui_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.asr_thread = None

        self.history = []

        self.status_var = tk.StringVar(value="未启动")
        self.partial_var = tk.StringVar(value="")
        self.trigger_var = tk.StringVar(value="尚未触发")

        self.wake_enabled_var = tk.BooleanVar(value=True)
        self.wake_word_var = tk.StringVar(value="醒来吧亲爱的")
        self.target_url_var = tk.StringVar(value="https://www.example.com")

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
        tk.Label(status_frame, textvariable=self.status_var, font=("Microsoft YaHei", 11), fg="blue").pack(side="left")

        trigger_frame = tk.Frame(self.root)
        trigger_frame.pack(fill="x", padx=10, pady=(6, 0))

        tk.Label(trigger_frame, text="网页唤醒：", font=("Microsoft YaHei", 11)).pack(side="left")
        tk.Label(trigger_frame, textvariable=self.trigger_var, font=("Microsoft YaHei", 11), fg="#b71c1c").pack(side="left")

        path_frame = tk.LabelFrame(self.root, text="当前资源路径", padx=8, pady=8)
        path_frame.pack(fill="x", padx=10, pady=10)

        path_text = (
            f"APP_DIR = {APP_DIR}\n"
            f"BUNDLE_DIR = {BUNDLE_DIR}\n"
            f"MODEL_DIR = {MODEL_DIR}\n"
            f"GPU_DLL_DIR = {GPU_DLL_DIR if GPU_DLL_DIR else '未找到'}"
        )
        tk.Label(
            path_frame,
            text=path_text,
            justify="left",
            anchor="w",
            font=("Consolas", 9)
        ).pack(fill="x")

        wake_frame = tk.LabelFrame(self.root, text="语音命令设置（修改后请重新开始识别）", padx=8, pady=8)
        wake_frame.pack(fill="x", padx=10, pady=10)

        row1 = tk.Frame(wake_frame)
        row1.pack(fill="x", pady=4)

        tk.Checkbutton(
            row1,
            text="启用语音唤醒打开网页",
            variable=self.wake_enabled_var,
            font=("Microsoft YaHei", 10)
        ).pack(side="left")

        row2 = tk.Frame(wake_frame)
        row2.pack(fill="x", pady=4)

        tk.Label(row2, text="唤醒词：", width=10, anchor="e", font=("Microsoft YaHei", 10)).pack(side="left")
        tk.Entry(row2, textvariable=self.wake_word_var, font=("Microsoft YaHei", 10)).pack(side="left", fill="x", expand=True)

        row3 = tk.Frame(wake_frame)
        row3.pack(fill="x", pady=4)

        tk.Label(row3, text="目标网址：", width=10, anchor="e", font=("Microsoft YaHei", 10)).pack(side="left")
        tk.Entry(row3, textvariable=self.target_url_var, font=("Microsoft YaHei", 10)).pack(side="left", fill="x", expand=True)

        tip = (
            "当前逻辑：持续识别中，一旦识别文本包含唤醒词，就自动打开目标网址。\n"
            "打包后请保证 exe 同级目录下存在 models 和 gpu_dlls。"
        )
        tk.Label(
            wake_frame,
            text=tip,
            justify="left",
            anchor="w",
            fg="#555555",
            font=("Microsoft YaHei", 9)
        ).pack(fill="x", pady=(6, 0))

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
            wraplength=1020,
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

        wake_word = self.wake_word_var.get().strip()
        target_url = self.target_url_var.get().strip()

        if self.wake_enabled_var.get():
            if not wake_word:
                messagebox.showwarning("提示", "启用网页唤醒时，唤醒词不能为空。")
                return
            if not target_url:
                messagebox.showwarning("提示", "启用网页唤醒时，目标网址不能为空。")
                return
            if not (target_url.startswith("http://") or target_url.startswith("https://")):
                messagebox.showwarning("提示", "目标网址建议以 http:// 或 https:// 开头。")
                return

        self.stop_event = threading.Event()
        self.asr_thread = ASRThread(
            ui_queue=self.ui_queue,
            stop_event=self.stop_event,
            wake_enabled=self.wake_enabled_var.get(),
            wake_word=wake_word,
            target_url=target_url,
        )
        self.asr_thread.start()
        self.status_var.set("启动中...")
        self.trigger_var.set("尚未触发")

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
        self.trigger_var.set("尚未触发")

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
            elif kind == "trigger":
                self.trigger_var.set(value)

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