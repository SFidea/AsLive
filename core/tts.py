"""
TTS：基于 Kokoro ONNX 的中文语音合成
"""
import numpy as np
import time
from pathlib import Path
from typing import Optional
import config

class TTSWrapper:
    def __init__(self, tts_config: dict = None):
        self.kokoro_paths = tts_config if tts_config else config.TTS_CONFIG
        
        # 验证文件存在
        for key, path in self.kokoro_paths.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Kokoro resource not found: {path}")
        
        # 懒加载模型
        self._kokoro_model = None
        self._g2p_model = None
        
        print("正在加载Kokoro TTS模型...")
        self._load_models()
        print("Kokoro TTS模型加载完成！")
    
    def _load_models(self):
        """加载 Kokoro 和 G2P 模型"""
        from misaki import zh
        from kokoro_onnx import Kokoro
        
        self._g2p_model = zh.ZHG2P(version="1.1")
        self._kokoro_model = Kokoro(
            model_path=self.kokoro_paths["model"],
            voices_path=self.kokoro_paths["voice"],
            vocab_config=self.kokoro_paths["vocab"]
        )
    
    @property
    def sample_rate(self) -> int:
        """返回采样率"""
        return 24000  # Kokoro 默认采样率
        
    def synthesize(self, text: str, speak_id: str = "zf_001", speed: float = 1.0):
        """
        合成文本为音频数据
        Returns:
            samples, sample_rate, latency
        """
        t1 = time.time()
        phonemes, _ = self._g2p_model(text)
        samples, sample_rate = self._kokoro_model.create(
            phonemes,
            voice=speak_id,
            speed=speed,
            is_phonemes=True
        )
        latency = time.time() - t1
        return samples, sample_rate, latency

if __name__ == "__main__":
    tts = TTSWrapper()
    print("TTS 初始化完成")