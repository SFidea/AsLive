import torch
import numpy as np

class VADWrapper:
    def __init__(self, repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False):
        print("正在初始化 Silero VAD 模型...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model,
            force_reload=force_reload,
            onnx=onnx,
            trust_repo=True
        )
        print("VAD 模型初始化完成！")
        
    def detect(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """
        检测给定音频块的人声概率
        audio_chunk: float32 numpy array, 建议长度为 512 (对于16000Hz)
        Returns: 0.0 ~ 1.0 的人声概率
        """
        tensor_chunk = torch.from_numpy(audio_chunk)
        prob = self.model(tensor_chunk, sample_rate).item()
        return prob
