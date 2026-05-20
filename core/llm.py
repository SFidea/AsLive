import ollama
import config
import time
from typing import Generator

class LLMWrapper:
    def __init__(self, model_name=config.LLM_MODEL_NAME):
        print(f"正在初始化Ollama LLM模型: {model_name}...")
        self.model_name = model_name
        self.client = ollama.AsyncClient()
        
        # 从 Soul.md 加载 System Prompt
        soul_path = config.BASE_DIR / "Soul.md"
        try:
            with open(soul_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
                print(f"已加载人设: {soul_path}")
        except FileNotFoundError:
            print(f"警告: 未找到人设文件 {soul_path}，使用默认空设定")
            self.system_prompt = "你是我的语音助手。"
            
        self.first_token_latency = 0.0

    async def warmup(self):
        """预热模型，将其加载到显存中"""
        print(f"正在预热 Ollama 模型 ({self.model_name})...")
        start_time = time.time()
        try:
            # 发送一个空请求来触发加载
            await self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': 'hi'}],
                options={'num_predict': 1}
            )
            print(f"Ollama 模型预热完成！耗时: {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"Ollama 模型预热失败: {e}")

    async def inference_stream_chat(self, messages: list, max_new_tokens: int = 500):
        """
        流式（多轮）：使用 Ollama 的 chat 接口 handle 多轮对话
        """
        start_time = time.time()
        first_token_received = False
        full_messages = [{'role': 'system', 'content': self.system_prompt}]
        full_messages.extend(messages)
        
        try:
            stream = await self.client.chat(
                model=self.model_name,
                messages=full_messages,
                stream=True,
                options={
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'num_predict': max_new_tokens,
                    'num_ctx': 4096,
                }
            )
            async for chunk in stream:
                if not first_token_received:
                    self.first_token_latency = time.time() - start_time
                    print(f"[LLM] Chat First Token Latency: {self.first_token_latency:.2f}s")
                    first_token_received = True
                yield chunk['message']['content']
        except Exception as e:
            print(f"LLM Chat Stream Error: {e}")
            yield "对不起，我现在有点不舒服。"

# ======================
# 使用示例
# ======================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        llm = LLMWrapper()
        await llm.warmup()
        
        print("--- 流式对话测试 ---")
        messages = [{'role': 'user', 'content': '我想去洗澡，但是水温好像不太对。'}]
        print("实时输出: ", end="", flush=True)
        async for chunk in llm.inference_stream_chat(messages):
            print(chunk, end="", flush=True)
        print("\n生成结束")

    asyncio.run(main())