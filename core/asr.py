from funasr import AutoModel
import numpy as np
import time


class ASRWrapper:
    
    def __init__(self, model_dir="paraformer-zh-streaming", device="mps"):
        print("正在初始化流式 ASR 模型 (paraformer-zh-streaming)...")
        self.chunk_size = [0, 10, 5]  # 600ms
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1
        
        self.model = AutoModel(
            model=model_dir,
            model_revision="v2.0.4",
            device=device,
            disable_pbar=True,
            disable_update=True
        )

    def transcribe(self, audio_input, cache=None, is_final=False):
        """
        流式 ASR 识别 - PCM 直连 (无需 ffmpeg)
        audio_input: np.ndarray float32, 16kHz 单声道 PCM
        cache: 外部维护的状态字典:
               cache["model_cache"]    -> FunASR 模型内部状态
               cache["last_pcm_length"] -> 游标
               cache["leftover_pcm"]    -> 残留样本
        """
        start = time.time()
        if cache is None:
            cache = {}
        if "model_cache" not in cache:
            cache["model_cache"] = {}
            
        try:
            full_pcm = audio_input
            
            # 游标机制：只处理新增样本
            last_processed = cache.get("last_pcm_length", 0)
            if len(full_pcm) <= last_processed:
                return "", time.time() - start
            
            new_samples = full_pcm[last_processed:]
            cache["last_pcm_length"] = len(full_pcm)
            
            # 与上次残留样本拼接
            leftover = cache.get("leftover_pcm", np.array([], dtype=np.float32))
            samples = np.concatenate([leftover, new_samples])
            
            chunk_stride = self.chunk_size[1] * 960  # 9600 样本 = 600ms
            
            if len(samples) == 0:
                return "", time.time() - start
            
            text = ""
            total = len(samples)
            i = 0
            
            while i + chunk_stride <= total:
                chunk = samples[i:i + chunk_stride]
                chunk_final = is_final and (i + chunk_stride * 2 > total)
                
                res = self.model.generate(
                    input=chunk,
                    cache=cache["model_cache"],
                    is_final=chunk_final,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back
                )
                if res and len(res) > 0 and "text" in res[0]:
                    text += res[0]["text"]
                i += chunk_stride
            
            # 处理 is_final 时的尾巴
            if is_final and i < total:
                res = self.model.generate(
                    input=samples[i:],
                    cache=cache["model_cache"],
                    is_final=True,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back
                )
                if res and len(res) > 0 and "text" in res[0]:
                    text += res[0].get("text", "")
                cache["leftover_pcm"] = np.array([], dtype=np.float32)
            else:
                cache["leftover_pcm"] = samples[i:]
            
        except Exception as e:
            import traceback
            print(f"[ASR] 识别异常: {e}")
            traceback.print_exc()
            return "", 0.0
        
        return text, time.time() - start