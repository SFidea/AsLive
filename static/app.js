// --- 1. THREE.JS SCENE ---
let scene, camera, renderer, sphere, count, originalPositions;

function init3D() {
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x030303, 0.002);

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 35;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Particle Sphere
    const geometry = new THREE.SphereGeometry(12, 128, 128);
    const material = new THREE.PointsMaterial({
        size: 0.15,
        color: 0xffffff,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending,
        vertexColors: true
    });

    count = geometry.attributes.position.count;
    const colors = [];
    const color1 = new THREE.Color(0x00f2ff); // Cyan
    const color2 = new THREE.Color(0xbd00ff); // Purple
    originalPositions = geometry.attributes.position.array.slice();

    for (let i = 0; i < count; i++) {
        const mixed = color1.clone().lerp(color2, Math.random());
        colors.push(mixed.r, mixed.g, mixed.b);
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    sphere = new THREE.Points(geometry, material);
    scene.add(sphere);
}

// --- 2. AUDIO & VISUALIZER ---
let audioContext, analyser, dataArray;
let micAnalyser, micDataArray; // 分离麦克风分析器
let isAudioInit = false;
let smoothedBass = 0;
let smoothedAvg = 0;

// Mini Monitor
const monitorCanvas = document.getElementById('mini-monitor');
const monitorCtx = monitorCanvas.getContext('2d');

function initAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // 播放用的分析器 (连接到 Destination)
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.8;
        dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        // 麦克风用的分析器 (不连接到 Destination)
        micAnalyser = audioContext.createAnalyser();
        micAnalyser.fftSize = 256;
        micAnalyser.smoothingTimeConstant = 0.8;
        micDataArray = new Uint8Array(micAnalyser.frequencyBinCount);
        
        isAudioInit = true;
        
        // ====== 直接连接到 destination，让浏览器接管标准 AEC 流同步 ======
        analyser.connect(audioContext.destination);
    }
    if (audioContext.state === 'suspended') {
        audioContext.resume();
    }
}

// --- 3. LOGIC & INTERACTION ---
const recordBtn = document.getElementById('record-btn');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const activeMessageZone = document.getElementById('active-message-zone');
const statusDisplay = document.getElementById('status-display');
const latencyDisplay = document.getElementById('latency-display');
const clearHistoryBtn = document.getElementById('clear-history-btn');
const audioPlayer = document.getElementById('audio-player');

// 录音相关
let isRecording = false;
let currentSessionId = null; // 当前正在进行的会话ID
let ws = null;
let isWsConnected = false;

// --- WebSocket 初始化 ---
function initWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log("WebSocket connected");
        isWsConnected = true;
        setStatus('READY');
    };
    
    ws.onclose = () => {
        console.log("WebSocket disconnected");
        isWsConnected = false;
        setStatus('DISCONNECTED');
        // 3秒后尝试重连
        setTimeout(initWebSocket, 3000);
    };
    
    ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        isWsConnected = false;
    };
    
    ws.onmessage = (event) => {
        handleWsMessage(event.data);
    };
}

// 处理 WS 消息
function handleWsMessage(dataStr) {
    try {
        const data = JSON.parse(dataStr);
        
        if (data.type === 'start') {
            setStatus('PROCESSING...', 'busy');
            currentSessionId = data.session_id;
            assistantMsgEl = null;
            fullContent = "";
        } else if (data.type === 'asr') {
            console.log("WS ASR Result:", data.text);
            // 实时识别出结果时本地流式拼接展示
            handleAsrChunk(data.text);
            lastAsrTime = Date.now();
        } else if (data.type === 'interrupt') {
            console.log("Receive Interrupt Signal from Server");
            finalizeInterrupt(); // 瞬间静音并清空队列
        } else if (data.type === 'text') {
            handleTextChunk(data.content);
        } else if (data.type === 'audio') {
            if (data.session_id && data.session_id !== currentSessionId) {
                console.log("Ignored stale audio chunk from previous response");
                return;
            }
            handleAudioChunk(data);
        } else if (data.type === 'end') {
            const assistantMsgEl = document.querySelector('.active-bubble.assistant.typing');
            if (assistantMsgEl) assistantMsgEl.classList.remove('typing');
            setStatus('READY');
        } else if (data.type === 'error') {
            console.error("Server Error:", data.message);
            setStatus('ERROR');
        }
    } catch (e) {
        console.error("Failed to parse WS message:", e);
    }
}

let assistantMsgEl = null;
let fullContent = "";

let userMsgEl = null;
let fullUserContent = "";

function handleAsrChunk(text) {
    if (!userMsgEl) {
        userMsgEl = addMessage('user', '');
    }
    fullUserContent += text;
    const contentDiv = userMsgEl.querySelector('.content');
    contentDiv.textContent = fullUserContent;
}

function handleTextChunk(content) {
    if (!assistantMsgEl) {
        assistantMsgEl = addMessage('assistant', '');
    }
    fullContent += content;
    const contentDiv = assistantMsgEl.querySelector('.content');
    contentDiv.textContent = fullContent;
}

let firstAudioReported = false;

function decodeBase64ToFloat32Array(base64) {
    const binary_string = window.atob(base64);
    const len = binary_string.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return new Float32Array(bytes.buffer);
}

function handleAudioChunk(data) {
    if (!firstAudioReported) {
        updateLatency(latencyStartTime);
        firstAudioReported = true;
    }
    
    const float32Data = decodeBase64ToFloat32Array(data.audio_data);
    audioQueue.push({
        buffer: float32Data,
        sampleRate: data.sample_rate || 24000,
        text: data.text
    });
    
    if (!isPlaying) {
        playNextAudio();
    }
}

// 页面加载即初始化
initWebSocket();

// 录音相关
let currentMicStream = null;
let pcmWorklet = null; // AudioWorklet 节点
let workletModuleLoaded = false;

async function startRecording() {
    if (isRecording) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: { 
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: false, // 关闭自动增益，保持麦克风音量平稳
                channelCount: 1
            } 
        });
        currentMicStream = stream;
        
        // 发送启动指令
        if (isWsConnected) {
            ws.send(JSON.stringify({ type: 'control', action: 'start' }));
        }
        
        isRecording = true;
        recordBtn.classList.add('recording');
        setStatus('LISTENING...', 'busy');
        
        // 连接可视化
        initAudioContext();
        
        const hpf = audioContext.createBiquadFilter();
        hpf.type = 'highpass';
        hpf.frequency.value = 80; // 从 200 改为 80，保留人声低频能量
        
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(hpf);
        hpf.connect(micAnalyser); // 可视化
        
        // 用 AudioWorklet 直接捕获 16kHz Int16 PCM，跳过 WebM 编解码
        if (!workletModuleLoaded) {
            await audioContext.audioWorklet.addModule('/static/pcm-processor.js');
            workletModuleLoaded = true;
        }
        pcmWorklet = new AudioWorkletNode(audioContext, 'pcm-processor');
        pcmWorklet.port.onmessage = (e) => {
            if (isWsConnected && isRecording) {
                // 恢复无条件发包机制，允许随时 100% 全双工打断
                ws.send(e.data); // 发送原始 PCM Int16 二进制
            }
        };
        hpf.connect(pcmWorklet);
        // 连接到静音输出以保持 worklet 在音频图中活跃
        const silentGain = audioContext.createGain();
        silentGain.gain.value = 0;
        pcmWorklet.connect(silentGain);
        silentGain.connect(audioContext.destination);
        
    } catch (e) {
        console.error('Mic error:', e);
        alert('无法访问麦克风');
        recordBtn.classList.remove('active');
    }
}

// 停止录音函数
function stopRecording() {
    if (pcmWorklet) {
        pcmWorklet.disconnect();
        pcmWorklet = null;
    }
    isRecording = false;
    recordBtn.classList.remove('recording');
    
    // 释放麦克风
    if (currentMicStream) {
        currentMicStream.getTracks().forEach(track => track.stop());
        currentMicStream = null;
    }
    
    setStatus('READY');
}

// 音频播放队列
let audioQueue = [];
let isPlaying = false;
let currentAudioSource = null;

// 状态更新
function setStatus(status, type = 'normal') {
    statusDisplay.textContent = status;
    const dot = document.querySelector('.status-dot');
    if (dot) {
        dot.className = 'status-dot ' + (type === 'busy' ? 'busy' : 'healthy');
    }
}

// 正式确认打断：ASR 识别到有效文字，清空队列并停止旧生成
async function finalizeInterrupt() {
    console.log("Finalizing interruption (ASR confirmed text).");
    
    if (currentAudioSource) {
        const source = currentAudioSource;
        currentAudioSource = null;
        try {
            source.onended = null; // 防止停播触发下一首
            source.stop();
        } catch (e) {}
        source.disconnect();
    }
    
    audioQueue = []; // 彻底清空待播放队列
    isPlaying = false;
    
    // 如果有正在进行的后端生成，通知其停止
    if (currentSessionId) {
        currentSessionId = null; 
    }
    
    assistantMsgEl = null;
    fullContent = "";
    
    setStatus('INTERRUPTED');
}

// 消息展示
let lastAsrTime = 0;
let latencyStartTime = 0;

function addMessage(role, content) {
    // 沉浸式设计：右上角只保留一个活跃的气泡
    activeMessageZone.innerHTML = '';
    
    // 清除打架的缓存节点引用，强制界面只更新当前活跃角色
    if (role === 'assistant') {
        userMsgEl = null;
        fullUserContent = "";
    } else if (role === 'user') {
        assistantMsgEl = null;
        fullContent = "";
    }
    
    const div = document.createElement('div');
    div.className = `active-bubble ${role}`;
    
    div.innerHTML = `<div class="content">${content}</div>`;
    
    if (role === 'assistant' && !content) {
        div.classList.add('typing');
    }
    
    activeMessageZone.appendChild(div);
    return div;
}

function updateLatency(startTime) {
    const baseTime = lastAsrTime || startTime;
    if (baseTime > 0) {
        const latency = Date.now() - baseTime;
        latencyDisplay.textContent = `LATENCY: ${latency}ms`;
        lastAsrTime = 0; // 重置
    }
}

// 播放音频队列
async function playNextAudio() {
    if (audioQueue.length === 0) {
        isPlaying = false;
        setStatus('WAITING FOR INPUT...');
        return;
    }

    isPlaying = true;
    const audioData = audioQueue.shift();
    
    // 连接到 WebAudio 以驱动可视化
    initAudioContext();
    
    // 确保 AudioContext 在播放前是激活的
    if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    try {
        const audioBuffer = audioContext.createBuffer(1, audioData.buffer.length, audioData.sampleRate);
        audioBuffer.getChannelData(0).set(audioData.buffer);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        
        // 挂载到 analyser 以驱动球体跳动及频谱
        source.connect(analyser); 
        
        currentAudioSource = source;

        source.onended = () => {
            if (currentAudioSource === source) {
                currentAudioSource = null;
                playNextAudio();
            }
        };

        setStatus('SPEAKING...', 'busy');
        source.start(0);
    } catch (e) {
        console.error("Playback failed:", e);
        playNextAudio();
    }
}

// 修改文字发送支持 WebSocket 全双工
async function sendText() {
    const text = textInput.value.trim();
    if (!text || !isWsConnected) return;
    
    textInput.value = '';
    addMessage('user', text);
    setStatus('THINKING...', 'busy');
    latencyStartTime = Date.now();
    
    assistantMsgEl = null;
    fullContent = "";
    firstAudioReported = false;

    // 文字发送也触发正式打断
    finalizeInterrupt();
    
    ws.send(JSON.stringify({ type: 'chat', text: text }));
}

// 录音逻辑
if (navigator.mediaDevices) {
    recordBtn.addEventListener('click', () => {
        if (!isRecording) {
            recordBtn.classList.add('active');
            const label = recordBtn.querySelector('.btn-label');
            if (label) label.textContent = 'STOP RECORDING';
            startRecording();
        } else {
            recordBtn.classList.remove('active');
            const label = recordBtn.querySelector('.btn-label');
            if (label) label.textContent = 'TAP TO SPEAK';
            stopRecording();
        }
    });
}

// --- 4. ANIMATION LOOP ---
let time = 0;
let mouseX = 0, mouseY = 0;

document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX - window.innerWidth / 2) * 0.0005;
    mouseY = (e.clientY - window.innerHeight / 2) * 0.0005;
});

function lerp(start, end, amt) {
    return (1 - amt) * start + amt * end;
}

function drawMonitor(data) {
    monitorCtx.clearRect(0, 0, monitorCanvas.width, monitorCanvas.height);
    monitorCtx.fillStyle = '#00f2ff';
    
    const barWidth = 3;
    const gap = 1;
    const step = Math.floor(data.length / (monitorCanvas.width / (barWidth + gap)));

    for (let i = 0; i < monitorCanvas.width; i += (barWidth + gap)) {
        const dataIndex = Math.floor(i / (barWidth + gap)) * step;
        const value = data[dataIndex] || 0;
        const percent = value / 255;
        const barHeight = percent * monitorCanvas.height;

        monitorCtx.globalAlpha = 0.5 + (percent * 0.5);
        monitorCtx.fillRect(i, monitorCanvas.height - barHeight, barWidth, barHeight);
    }
}

function drawIdleMonitor(t) {
    monitorCtx.clearRect(0, 0, monitorCanvas.width, monitorCanvas.height);
    monitorCtx.fillStyle = 'rgba(255, 255, 255, 0.1)';
    for (let i = 0; i < monitorCanvas.width; i += 4) {
        const h = 5 + Math.sin(i * 0.1 + t * 5) * 3;
        monitorCtx.fillRect(i, monitorCanvas.height - h, 3, h);
    }
}

function animate() {
    requestAnimationFrame(animate);
    time += 0.005;

    let bassTarget = 0;
    let avgTarget = 0;
    let currentData = null;

    if (isAudioInit) {
        if (isRecording) {
            micAnalyser.getByteFrequencyData(micDataArray);
            currentData = micDataArray;
            
            // 同时获取当前正在播放的能量，用于回声掩蔽
            if (isPlaying) {
                analyser.getByteFrequencyData(dataArray);
            }
        } else {
            analyser.getByteFrequencyData(dataArray);
            currentData = dataArray;
        }

        const overallSum = currentData.reduce((a, b) => a + b, 0);
        avgTarget = overallSum / currentData.length;

        bassTarget = currentData[5] / 255;
        drawMonitor(currentData);
    } else {
        drawIdleMonitor(time);
    }

    smoothedBass = lerp(smoothedBass, bassTarget, 0.08);
    smoothedAvg = lerp(smoothedAvg, avgTarget / 255, 0.1);

    if (sphere) {
        const scaleTarget = 1 + (smoothedBass * 0.3);
        sphere.scale.lerp(new THREE.Vector3(scaleTarget, scaleTarget, scaleTarget), 0.05);

        const positions = sphere.geometry.attributes.position.array;
        const audioForce = smoothedAvg * 5.0;

        for (let i = 0; i < count; i++) {
            const px = originalPositions[i * 3];
            const py = originalPositions[i * 3 + 1];
            const pz = originalPositions[i * 3 + 2];

            let noise = Math.sin(px * 0.4 + time * 2) * 
                        Math.cos(py * 0.3 + time * 1.5) * 
                        Math.sin(pz * 0.4 + time * 2.5);

            const displacement = 1 + (noise * 0.1) + (noise * audioForce * 0.25);

            positions[i * 3]     = px * displacement;
            positions[i * 3 + 1] = py * displacement;
            positions[i * 3 + 2] = pz * displacement;
        }
        sphere.geometry.attributes.position.needsUpdate = true;
        sphere.rotation.y += 0.001 + (smoothedAvg * 0.002);
        sphere.rotation.x += (mouseY - sphere.rotation.x) * 0.05;
        sphere.rotation.y += (mouseX - sphere.rotation.y) * 0.05;
    }

    renderer.render(scene, camera);
}


// 初始化
try {
    console.log("Initializing 3D Scene...");
    init3D();
    animate();
} catch (e) {
    console.error("3D Init Failed:", e);
    // 即使 3D 失败，也要让聊天可用
}

console.log("App loaded, binding events...");

// 事件绑定
if (sendBtn) {
    sendBtn.addEventListener('click', () => {
        console.log("Send button clicked");
        sendText();
    });
} else {
    console.error("Send button not found!");
}

if (textInput) {
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            console.log("Enter key pressed");
            sendText();
        }
    });
}

if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', async () => {
        await fetch('/history', { method: 'DELETE' });
        activeMessageZone.innerHTML = '';
        setStatus('HISTORY CLEARED');
    });
}

window.addEventListener('resize', () => {
    if (camera && renderer) {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
});
