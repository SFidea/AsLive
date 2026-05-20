# 🎙️ AsLive - 沉浸式实时语音交互系统

<div align="center">

![AsLive](static/AsLive.png)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)](https://ollama.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-TTS_Acceleration-blue?style=for-the-badge)](https://onnxruntime.ai/)

</div>

---

**AsLive** 是一款基于 Web 的、高品质且极具沉浸感的实时语音交互助手系统。它融合了流式语音识别 (ASR)、本地大语言模型 (LLM) 和超自然流式语音合成 (TTS)，并通过 WebSocket 全双工通信实现**支持随时插话打断**的极致对话体验。前端配有由 Three.js 驱动的璀璨 3D 粒子星云视觉动效，能够根据音频振幅与频率进行实时形变与律动。

---

## ✨ 核心特性

- ⚡ **极致超低延迟**：
  - 基于 WebSocket 全双工长连接，实现真正的双向音频流传输。
  - LLM 生成与 TTS 合成以“流式句级流水线 (Sentence-level Pipeline)”并行调度，端到端首音延迟缩短至极致，带来真人对话般的流畅度。
- 🎙️ **智能双向打断 (随时插话)**：
  - 集成 **Silero VAD**，实时在前端与后端检测用户发音。
  - 一旦检测到用户说话，系统会瞬间下发打断信号，**毫秒级静音**当前正在播放的助理音频并清空合成队列，极速响应用户的最新指令。
- 🔮 **沉浸式 3D 粒子星云**：
  - 基于 Three.js 的 3D 粒子球体交互界面，呈现深邃的科技美学。
  - 视觉效果采用双模态动态融合：用户录音时，星云随麦克风声波振幅起伏；助手播报时，粒子星云以澎湃的音频频域能量进行形变和波浪式起伏。
- 🧠 **本地双脑驱动**：
  - **ASR（语音识别）**：引入阿里 FunASR 框架下的 `paraformer-zh-streaming` 工业级流式识别模型，带来极速、精准的语音转文字体验。
  - **LLM（大语言模型）**：完美适配本地 Ollama 框架，默认使用 `llama3.2:3b`（或自定义模型如 `qwen2.5` 等），通过 SSE 极速吐字。
  - **TTS（语音合成）**：集成 ONNX 加速的 Kokoro-ONNX (v1.1-zh)，结合 misaki G2P 前端，提供宛如真人的中文自然声线。
- 🎭 **具有“灵魂”的人设系统**：
  - 系统启动时会自动加载根目录下的 `Soul.md` 文件作为全局人设（System Prompt）。你可以通过修改此文件随时赋予她独特的个性、说话语气及世界观。

---

## 📂 项目结构

```txt
AsLive/
├── checkpoints/              # 模型权重文件夹 (放置 Kokoro 模型)
│   └── kokoro/
│       ├── kokoro-v1.1-zh.onnx  # TTS 核心模型
│       ├── voices-v1.1-zh.bin   # 声音配置文件
│       └── config-v1.1-zh.json  # 字典与词表配置
├── core/                     # 核心模型封装组件
│   ├── asr.py                # FunASR Streaming ASR 组件
│   ├── llm.py                # Ollama Async LLM 组件 (自动加载 Soul.md)
│   ├── tts.py                # Kokoro ONNX 语音合成封装
│   └── vad.py                # Silero VAD 语音激活检测组件
├── static/                   # 极简炫酷的前端界面
│   ├── index.html            # Web 入口
│   ├── style.css             # 沉浸式毛玻璃与暗黑风样式
│   ├── app.js                # Three.js 动效、AudioContext 捕获与 WebSocket 通信逻辑
│   └── pcm-processor.js      # AudioWorklet PCM 采集处理器 (高效捕获 16kHz Int16 原始流)
├── outputs/                  # 临时交互日志与音频输出
├── api_server.py             # FastAPI 后端主服务器 (处理全双工 WebSocket 与业务调度)
├── config.py                 # 全局路径及超参数配置文件
├── Soul.md                   # 助手的“人设记忆”文件 (系统提示词)
└── requirements.txt          # Python 依赖清单
```

---

## 🚀 快速开始

### 1. 环境准备

确保您的操作系统上安装了 Python 3.10+，并具备 CUDA/MPS 硬件加速环境（若无，CPU 亦可运行）。

```bash
# 克隆/进入项目目录并安装依赖
pip install -r requirements.txt
```

> [!NOTE]
> 项目依赖 `onnxruntime` 进行 TTS 推理。如果您的设备支持 GPU 加速，建议根据硬件平台安装 `onnxruntime-gpu`。

### 2. 部署本地大模型 (Ollama)

1. 安装 [Ollama](https://ollama.com/) 并确保其在后台运行。
2. 下载默认的 LLM 模型：
   ```bash
   ollama pull llama3.2:3b
   ```
   *(您也可以在 `config.py` 中将 `LLM_MODEL_NAME` 修改为您本地的其他模型，例如 `qwen2.5:7b` 或自定义模型)*

### 3. 下载并配置 Kokoro TTS 模型 🌟

本项目使用 Kokoro-ONNX 进行极其自然的本地语音合成。为了系统能正常运行，**您必须手动下载模型文件并将其放入 `checkpoints` 文件夹中**。

#### **具体步骤：**
1. 在项目根目录下，确保已创建 `checkpoints/kokoro` 目录。
2. 前往 GitHub Releases 或 Hugging Face 下载以下三个核心文件：
   - **模型文件**：`kokoro-v1.1-zh.onnx`
   - **声音特征库**：`voices-v1.1-zh.bin`
   - **字典词表配置**：`config-v1.1-zh.json`
3. 将下载的 3 个文件放置到 **`checkpoints/kokoro/`** 目录中。

配置完成后，请检查路径是否如下所示：
```txt
checkpoints/
└── kokoro/
    ├── config-v1.1-zh.json
    ├── kokoro-v1.1-zh.onnx
    └── voices-v1.1-zh.bin
```

### 4. 启动语音助手

```bash
python api_server.py
```
运行后，后端服务将启动在：`http://localhost:8000`

### 5. 体验沉浸式交互

1. 在浏览器中访问 `http://localhost:8000/`。
2. 页面成功加载后，您将看到深邃的 3D 粒子星云在中心流转。
3. **文字输入**：在底部左下角的输入框中输入内容，敲击回车或点击发送按钮即可与助手文字交互。
4. **语音对话**：
   - 点击底部的 **`TAP TO SPEAK`** 按钮（图标变红），开始对麦克风说话。
   - 说话期间，中控粒子球会随您的声音波动。
   - **静音检测自动触发**：当您说完并停顿超过设定时间（默认 `350ms`），后端 VAD 会自动判定您说话结束并进入处理，您无需点击停止。
   - **打断功能**：在助理正在说话播放音频时，您直接点击麦克风或打字，系统会瞬间阻断当前播放并进入全新对话。

---

## 🎭 塑造个性

在项目根目录的 `Soul.md` 中，定义了助手的性格、语气偏好和核心原则。
您可以随意编辑该文件，例如：
* 将她改为带有御姐、萝莉或专业职场人的人设；
* 控制她第一句开口的客套语；
* 规范其回答的边界条件和安全规则。

---

## ⚡ 性能与优化指标

- **ASR 延迟**：使用 FunASR 增量流式识别，单次包捕获仅 `600ms`。
- **LLM 首字延迟**：Ollama 本地显卡/M系列芯片加速下首 Token 生成在 `0.2s - 0.5s` 左右。
- **TTS 耗时**：使用 ONNX Runtime 加速，首句合成耗时仅约 `100ms - 200ms`。
- **端到端首音延迟**：在 M系列 Mac / RTX 显卡上，从用户停顿到听到助手首个单词播放，综合耗时大约在 **1.5s - 2s** 内，体验极佳。
