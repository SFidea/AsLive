/**
 * AudioWorklet Processor: 捕获麦克风 PCM 并降采样到 16kHz Int16
 * 直接发送原始 PCM 给后端，彻底跳过 WebM 编码/ffmpeg 解码
 */
class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.ratio = sampleRate / 16000;       // 降采样比例 (e.g. 48000/16000 = 3)
        this.srcPos = 0;                        // 降采样累加器
        this.chunkSize = 800;                   // 每次发送 800 个 16kHz 样本 = 50ms
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || input.length === 0 || !input[0]) return true;

        const src = input[0]; // 单声道

        // 通过累加器进行降采样 (Bresenham 式) 配合低通平滑处理
        for (let i = 0; i < src.length; i++) {
            // 简单的低频平滑 (避免高频尖锐噪声折叠)。使用过去的样本进行简单移动平均
            const smoothed = (src[i] + (input[0][i-1] || 0) + (input[0][i-2] || 0)) / 3.0;

            this.srcPos += 16000;
            if (this.srcPos >= sampleRate) {
                this.srcPos -= sampleRate;
                this.buffer.push(smoothed);
            }
        }

        // 攒够一个 chunk 就发送
        while (this.buffer.length >= this.chunkSize) {
            const chunk = this.buffer.splice(0, this.chunkSize);
            const int16 = new Int16Array(chunk.length);
            for (let j = 0; j < chunk.length; j++) {
                const s = Math.max(-1, Math.min(1, chunk[j]));
                int16[j] = s < 0 ? s * 32768 : s * 32767;
            }
            this.port.postMessage(int16.buffer, [int16.buffer]);
        }

        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
