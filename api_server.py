"""
语音助手 Web API 服务器
基于 FastAPI，提供 SSE 流式响应
"""
import asyncio
import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import logging
from logging.handlers import RotatingFileHandler

# 对话历史
conversation_history = []

# 句子分隔符
SENTENCE_DELIMITERS = set(',，。！？；.!?;\n')

# 导入核心模块
from core.asr import ASRWrapper
from core.llm import LLMWrapper
from core.tts import TTSWrapper
from core.vad import VADWrapper
import config
import re
import time
app = FastAPI(title="Voice Assistant API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件与输出目录
STATIC_DIR = config.STATIC_DIR
OUTPUT_DIR = config.OUTPUT_DIR

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# === 后台日志系统配置 ===
log_file = OUTPUT_DIR / "interaction.log"
logger = logging.getLogger("VoiceAssistant")
logger.setLevel(logging.INFO)

# 文件输出 (最大 5MB, 循环3个文件)
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台输出
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
# ========================

# 全局模型实例（懒加载）
_models = {
    "asr": None,
    "llm": None,
    "tts": None,
    "vad": None,
    "initialized": False
}



def get_models():
    """懒加载模型"""
    if not _models["initialized"]:
        print("正在初始化模型...")
        # 现在 Wrapper 会自动使用 config.py 中的默认值
        _models["asr"] = ASRWrapper()
        _models["llm"] = LLMWrapper()
        _models["tts"] = TTSWrapper()
        _models["vad"] = VADWrapper()
        _models["initialized"] = True
        print("模型初始化完成！")
    return _models["asr"], _models["llm"], _models["tts"], _models["vad"]


def clean_text_for_tts(text: str) -> str:
    """清理文本用于 TTS"""
    # 如果包含对话格式，提取最后一个助手回复
    if '助手:' in text or '助手：' in text:
        parts = re.split(r'助手[：:]\s*', text)
        if len(parts) > 1:
            last_response = parts[-1]
            last_response = re.split(r'\n\s*用户[：:]', last_response)[0]
            text = last_response
    # 移除角色前缀
    text = re.sub(r'^(用户|助手|User|Assistant)[：:]\s*', '', text, flags=re.MULTILINE)
    return text.strip()



async def process_streaming(llm_stream, speak_id: str = "zf_001", speed: float = 1.0, 
                            asr_latency: float = 0.0, user_speech_start_time: float = 0.0):
    """
    流式处理：LLM 生成 + TTS 合成 (流水线并发优化版)
    """
    global conversation_history
    
    asr, llm, tts, _ = get_models()
    
    buffer = ""
    full_response = ""
    
    # 任务队列，存储 (task, clean_text)
    # 使用 list 模拟队列，保持顺序
    tts_tasks = [] 
    audio_index = 0

    in_think_tag = False
    process_start_time = time.time()
    llm_end_time = None
    first_tts_reported = False
    first_audio_time = None
    first_text_segment = ""
    first_tts_latency = 0
    session_id = str(uuid.uuid4())[:8]
    
    # 发送开始事件
    yield {"type": "start", "session_id": session_id}
    
    # 定义异步 TTS 函数
    async def synthesize_worker(text, index):
        try:
            # 运行在线程池中
            loop = asyncio.get_running_loop()
            t0 = time.time()
            
            # 1. 文本转语音 (CPU密集型，放入线程池)
            samples, sample_rate, _ = await loop.run_in_executor(
                None, 
                lambda: tts.synthesize(text, speak_id, speed)
            )
            
            # 2. 直接转换为 base64 发送 (不再写磁盘)
            import base64
            audio_bytes = samples.astype(np.float32).tobytes()
            b64_data = base64.b64encode(audio_bytes).decode('utf-8')
            
            latency = time.time() - t0
            return b64_data, sample_rate, latency, text
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return None, 0, 0.0, text

    # 用于检查并发送已完成的 TTS 任务
    async def check_and_yield_tts():
        nonlocal first_tts_reported, first_audio_time, first_text_segment, first_tts_latency
        # 只要队列头部的任务完成了，就发送
        while tts_tasks and tts_tasks[0].done():
            task = tts_tasks.pop(0)
            try:
                b64_data, sample_rate, tts_latency, text_segment = await task
                if b64_data:
                    # 性能统计
                    if not first_tts_reported:
                        first_tts_reported = True
                        first_audio_time = time.time()
                        first_text_segment = text_segment
                        first_tts_latency = tts_latency
                    
                    yield {"type": "audio", "audio_data": b64_data, "sample_rate": sample_rate, "text": text_segment, "session_id": session_id}
            except Exception as e:
                logger.error(f"Error yielding TTS result: {e}")

    try:
        # 流式生成
        async for chunk in llm_stream:
            
            full_response += chunk
            
            # 处理 <think> 标签
            if '<think>' in chunk:
                in_think_tag = True
                print("\n[LLM is Thinking] ", end="", flush=True)
                before_think = chunk.split('<think>')[0]
                if before_think:
                    buffer += before_think
                    yield {"type": "text", "content": before_think}
                continue
            
            if '</think>' in chunk:
                in_think_tag = False
                print(" [Done]\n")
                after_think = chunk.split('</think>')[-1]
                if after_think:
                    buffer += after_think
                else:
                    continue
                
            if in_think_tag:
                print(".", end="", flush=True)
                continue
                
            buffer += chunk
            
            yield {"type": "text", "content": chunk}
            
            while True:
                delimiter_pos = -1
                for i, char in enumerate(buffer):
                    if char in SENTENCE_DELIMITERS:
                        delimiter_pos = i
                        break
                
                if delimiter_pos == -1:
                    break
                
                sentence = buffer[:delimiter_pos + 1]
                buffer = buffer[delimiter_pos + 1:]
                
                clean_text = clean_text_for_tts(sentence)
                if clean_text:
                    task = asyncio.create_task(synthesize_worker(clean_text, audio_index))
                    tts_tasks.append(task)
                    audio_index += 1
            
            async for event in check_and_yield_tts():
                yield event
        
        # LLM 生成结束
        llm_end_time = time.time()
        llm_total_time = llm_end_time - process_start_time

        if buffer.strip():
            clean_text = clean_text_for_tts(buffer)
            if clean_text:
                task = asyncio.create_task(synthesize_worker(clean_text, audio_index))
                tts_tasks.append(task)
                audio_index += 1
        
        while tts_tasks:
            if not tts_tasks[0].done():
                await tts_tasks[0]
            async for event in check_and_yield_tts():
                yield event

        clean_response = clean_text_for_tts(full_response)
        if clean_response:
            # 这里的总延时计算逻辑：
            # 第一句音频准备好的时刻 - 用户真正开口说话的时刻
            total_e2e_from_start = (first_audio_time - user_speech_start_time) if first_audio_time and user_speech_start_time else 0
            logger.info(
                f"📊 [端到端性能报告] "
                f"🎤 ASR全程(含说话): {asr_latency:.3f}s | "
                f"🧠 LLM首字: {llm.first_token_latency:.3f}s | "
                f"🧠 LLM总成: {llm_total_time:.3f}s | "
                f"🗣️ TTS首句: {first_tts_latency:.3f}s | "
                f"⚡ 端到端总延时(从开口到出声): {total_e2e_from_start:.3f}s"
            )
            logger.info(f"🤖 [LLM 完整回复] {clean_response.strip()}")
            conversation_history.append({"role": "assistant", "content": clean_response.strip()})
        
        yield {"type": "end", "full_response": clean_response}

    except asyncio.CancelledError:
        print("[Streaming] 会话被打断，清理仍在运行的后台 TTS 任务释放内存...")
        for t in tts_tasks:
            if not t.done():
                t.cancel()
        raise




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    全双工 WebSocket：支持随时插话（打断）
    """
    await websocket.accept()
    print("WebSocket 全双工连接已启动")
    
    session_id = str(uuid.uuid4())[:8]
    active_response_task = None
    
    # 获取模型
    asr, llm, tts, vad_model = get_models()

    async def handle_response(user_text, asr_latency, user_speech_start_time, speak_id="zf_001", speed=1.0):
        """处理回复的任务包装器"""
        try:
            conversation_history.append({"role": "user", "content": user_text})
            llm_stream = llm.inference_stream_chat(conversation_history[-6:])
            async for event in process_streaming(llm_stream, speak_id, speed, asr_latency, user_speech_start_time):
                await websocket.send_json(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"Reply Error: {e}")

    # 统一管理用户说话起步时间
    user_speech_start_time = 0.0

    # ASR 持续监听与流式状态 (使用隔离的字典状态避免线程冲刷)
    utterance = {
        "text": "",
        "pcm_buffer": np.array([], dtype=np.float32),
        "asr_cache": {},
        "chunks_since_last_asr": 0,
        "asr_running": False
    }
    ASR_STRIDE = 1              # 每收到一个 PCM 包就检查一次，通过样本数平滑
    
    # VAD 状态初始化
    vad_buffer = np.array([], dtype=np.float32)
    consecutive_speech_chunks = 0
    consecutive_silence_chunks = 0
    has_spoken = False
    
    # 截断时间设置
    
    # 截断时间设置
    silence_chunks_target = round(config.VAD_SILENCE_THRESHOLD_MS / 32.0)  # 32ms 每块 (16000Hz 下的512 samples)

    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                current_time = time.time()
                # 前端 AudioWorklet 直接发送 16kHz Int16 PCM
                pcm_int16 = np.frombuffer(message["bytes"], dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                
                # 能量过滤：如果全包平均能量极低，直接跳过 VAD
                rms = np.sqrt(np.mean(pcm_float**2))
                
                utterance["pcm_buffer"] = np.concatenate([utterance["pcm_buffer"], pcm_float])
                vad_buffer = np.concatenate([vad_buffer, pcm_float])
                utterance["chunks_since_last_asr"] += 1
                
                # --- VAD 实时检测 (每 512 个样本 = 32ms) ---
                vad_triggered = False
                while len(vad_buffer) >= 512:
                    vad_chunk = vad_buffer[:512]
                    vad_buffer = vad_buffer[512:]
                    
                    # 只有能量超过基本阈值才跑 VAD，节省 CPU 并过滤底噪
                    
                    # 只有能量超过基本阈值才跑 VAD，节省 CPU 并过滤底噪
                    if rms > 0.005: 
                        prob = vad_model.detect(vad_chunk, 16000)
                    else:
                        prob = 0.0
                    
                    # 动态阈值：0.7
                    v_threshold = 0.7
                    s_chunks_threshold = 10 
                    
                    if prob > v_threshold:
                        consecutive_speech_chunks += 1
                        consecutive_silence_chunks = 0
                        
                        if consecutive_speech_chunks >= s_chunks_threshold:
                            if not has_spoken:
                                has_spoken = True
                                user_speech_start_time = time.time() # 记录用户开口时刻
                                logger.warning(f"⚠️ [VAD] 检测到用户声音")
                                if active_response_task and not active_response_task.done():
                                    active_response_task.cancel()
                                asyncio.create_task(websocket.send_json({"type": "interrupt"}))
                    else:
                        consecutive_silence_chunks += 1
                        consecutive_speech_chunks = 0
                        if has_spoken and consecutive_silence_chunks >= silence_chunks_target:
                            has_spoken = False
                            vad_triggered = True
                            vad_end_time = time.time() # 记录用户停顿确认的时间点
                            logger.info("🛑 [VAD] 检测到停顿，确认用户说话结束")
                            vad_buffer = np.array([], dtype=np.float32) # 截断后重置 VAD buffer
                            break
                            
                # --- VAD 触发判定用户说完一句话 ---
                if vad_triggered:
                    old_utterance = utterance
                    utterance = {
                        "text": "",
                        "pcm_buffer": np.array([], dtype=np.float32),
                        "asr_cache": {},
                        "chunks_since_last_asr": 0,
                        "asr_running": False
                    }
                    
                    async def finalize_and_trigger(utt, t_speech_start):
                        nonlocal active_response_task
                        
                        # 对剩余还没送进 ASR 的 PCM 做最后一次兜底识别
                        if len(utt["pcm_buffer"]) > 0:
                            wait_start = time.time()
                            while utt["asr_running"] and time.time() - wait_start < 0.5:
                               await asyncio.sleep(0.01)
                               
                            utt["asr_running"] = True
                            try:
                                # 强制最终识别，提高准确率
                                new_speech, _ = await asyncio.to_thread(
                                   asr.transcribe, audio_input=utt["pcm_buffer"], cache=utt["asr_cache"], is_final=True
                                )
                                if new_speech:
                                    utt["text"] += new_speech
                                    await websocket.send_json({"type": "asr", "text": new_speech})
                            except: pass
                            finally:
                                utt["asr_running"] = False
                                
                        if utt["text"].strip():
                            t_asr_end = time.time()
                            # ASR 全程：从开口到 LLM 准备好开始处理之前的总时间
                            total_asr_latency = t_asr_end - t_speech_start
                            logger.info(f"🎤 [用户 ASR 最终结果] {utt['text']}")
                            if active_response_task and not active_response_task.done():
                                active_response_task.cancel()
                            active_response_task = asyncio.create_task(handle_response(utt["text"], total_asr_latency, t_speech_start))

                    asyncio.create_task(finalize_and_trigger(old_utterance, user_speech_start_time))
                    continue 
                
                # --- ASR 增量识别逻辑 ---
                # ASR_STRIDE = 1 表示每个 200ms 包进来都检查，但 len >= 9600 保证了 600ms 的识别颗粒度
                if utterance["chunks_since_last_asr"] >= ASR_STRIDE and len(utterance["pcm_buffer"]) >= 9600 and not utterance["asr_running"]:
                    utterance["chunks_since_last_asr"] = 0
                    snapshot = utterance["pcm_buffer"].copy()
                    
                    async def continuous_listen(utt, data):
                        utt["asr_running"] = True
                        try:
                            # 增量识别不设置 is_final，保留长句上下文
                            new_speech, _ = await asyncio.to_thread(
                                asr.transcribe, audio_input=data, cache=utt["asr_cache"]
                            )
                            if not new_speech:
                                return
                                
                            utt["text"] += new_speech
                            await websocket.send_json({"type": "asr", "text": new_speech})
                        except Exception as e:
                            print(f"[WS ASR] 异常: {e}")
                        finally:
                            utt["asr_running"] = False

                    asyncio.create_task(continuous_listen(utterance, snapshot))
                
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    m_type = data.get("type")
                    if m_type == "control" and data.get("action") == "start":
                        utterance = {
                            "text": "",
                            "pcm_buffer": np.array([], dtype=np.float32),
                            "asr_cache": {},
                            "chunks_since_last_asr": 0,
                            "asr_running": False
                        }
                        if active_response_task: active_response_task.cancel()
                    elif m_type == "chat":
                        user_text = data.get("text", "")
                        if active_response_task: active_response_task.cancel()
                        # 文字聊天时，将开口时间设为当前，对应延时为0
                        now = time.time()
                        active_response_task = asyncio.create_task(handle_response(user_text, 0.0, now))
                except: pass

    except (WebSocketDisconnect, RuntimeError):
        print("WebSocket 断开")
    finally:
        if active_response_task:
            active_response_task.cancel()
        try:
            await websocket.close()
        except:
            pass


@app.get("/")
async def index():
    """返回主页"""
    return FileResponse(STATIC_DIR / "index.html")



@app.get("/history")
async def get_history():
    """获取对话历史"""
    return {"history": conversation_history}


@app.delete("/history")
async def clear_history():
    """清空对话历史"""
    global conversation_history
    conversation_history = []
    return {"status": "ok"}


@app.on_event("startup")
async def startup():
    """启动时预加载模型"""
    print("服务器启动中...")
    print("正在预加载模型，请稍候...")
    # 预加载模型，避免第一次请求卡顿
    asr, llm, tts, vad = get_models()
    # 手动预热 LLM
    await llm.warmup()
    print("服务器启动完成，模型已就绪！")


if __name__ == "__main__":
    import uvicorn
    # 开启热重载，方便开发调试
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
