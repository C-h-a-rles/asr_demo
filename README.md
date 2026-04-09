# Local ASR GUI

一个基于 **sherpa-onnx + SenseVoice + silero-vad + Tkinter** 的本地中文实时语音识别 GUI Demo，支持：

- 本地离线语音识别
- 滚动字幕式实时显示
- 历史字幕累计
- 语音唤醒打开网页
- GPU 优先推理，失败自动回退 CPU
- Windows 下 PyInstaller 打包运行

---

## 1. 项目功能

本项目是一个本地语音交互原型，主要提供两类能力：

### 实时语音识别
- 从麦克风采集语音
- 使用 VAD 自动切分语音段
- 在说话过程中持续刷新“当前滚动字幕”
- 在一句话结束后将结果固化到“已确认字幕”

### 简单语音命令
- 通过识别文本匹配唤醒词
- 识别到指定短语后自动打开浏览器并访问目标网址

默认示例：
- 唤醒词：`醒来吧亲爱的`
- 动作：打开指定网页

---

## 2. 技术栈

- **ASR**: sherpa-onnx
- **模型**: SenseVoice
- **VAD**: silero_vad.onnx
- **GUI**: Tkinter
- **音频采集**: sounddevice
- **推理后端**: onnxruntime / onnxruntime-gpu
- **打包**: PyInstaller

---

## 3. 运行环境

推荐环境：

- Windows 10 / 11
- Python 3.10
- NVIDIA GPU（可选，推荐）
- CUDA 12.x
- cuDNN 9.x
- 可正常使用的麦克风

说明：
- 项目支持 **GPU 优先，CPU 回退**
- 如果 GPU 运行时加载失败，程序会自动尝试使用 CPU

---

## 4. 项目结构

源码运行时建议目录如下：

```text
project_root/
├─ v1.py
├─ models/
│  ├─ silero_vad.onnx
│  └─ sensevoice/
│     ├─ model.int8.onnx
│     └─ tokens.txt
└─ README.md
```

