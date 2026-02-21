# Open-LLM-VTuber TTS Streaming 实现分析

## 概述

本文档分析 Open-LLM-VTuber 项目中 TTS (Text-to-Speech) 功能的实现方式，特别是与"streaming"相关的部分。

**核心结论**：当前实现采用**伪流式架构**——并非真正的实时音频流，而是通过**并行生成 + 有序交付**来优化延迟。

---

## 1. 架构设计

### 1.1 整体流程

```
用户输入 → LLM生成文本 → 句子分割 → TTS生成 → 音频打包 → WebSocket发送 → 前端播放
                    ↓
              并行处理多句TTS
                    ↓
              序列号保证顺序
```

### 1.2 关键组件

| 组件 | 文件路径 | 职责 |
|------|----------|------|
| `TTSInterface` | `src/open_llm_vtuber/tts/tts_interface.py` | TTS引擎抽象接口 |
| `TTSTaskManager` | `src/open_llm_vtuber/conversations/tts_manager.py` | 管理TTS任务并行执行与有序交付 |
| `prepare_audio_payload` | `src/open_llm_vtuber/utils/stream_audio.py` | 音频处理与WebSocket负载准备 |
| `TTSFactory` | `src/open_llm_vtuber/tts/tts_factory.py` | TTS引擎工厂，支持20+种引擎 |

---

## 2. TTS 接口设计

### 2.1 基础接口 (`TTSInterface`)

```python
class TTSInterface(metaclass=abc.ABCMeta):
    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """默认使用 asyncio.to_thread 在线程池中运行同步生成"""
        return await asyncio.to_thread(self.generate_audio, text, file_name_no_ext)

    @abc.abstractmethod
    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """生成音频文件，返回文件路径"""
        raise NotImplementedError
```

**关键特点**：
- 所有TTS引擎都返回**完整音频文件路径**，而非音频流
- `async_generate_audio` 提供默认的异步包装（线程池）
- 子类可以覆盖 `async_generate_audio` 实现真正的异步生成

### 2.2 支持的TTS引擎 (20+)

- **云端API**: Azure TTS, Edge TTS, OpenAI TTS, ElevenLabs, Cartesia, Minimax, SiliconFlow, Fish API
- **本地推理**: Bark, CosyVoice, CosyVoice2, MeloTTS, X-TTS, GPT-SoVITS, Coqui TTS, Spark TTS, Piper TTS, Sherpa-ONNX
- **系统TTS**: pyttsx3

---

## 3. 伪流式实现详解

### 3.1 核心问题

真正的流式TTS应该：
```
文本片段 → TTS引擎 → 音频块 → 立即发送 → 前端播放
```

当前实现的方式：
```
文本句子 → TTS引擎 → 完整音频文件 → 打包发送 → 前端播放
```

### 3.2 TTSTaskManager: 并行生成 + 有序交付

**文件**: `src/open_llm_vtuber/conversations/tts_manager.py`

```python
class TTSTaskManager:
    def __init__(self) -> None:
        self.task_list: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        self._payload_queue: asyncio.Queue[Dict] = asyncio.Queue()
        self._sender_task: Optional[asyncio.Task] = None
        self._sequence_counter = 0           # 分配序列号
        self._next_sequence_to_send = 0      # 下一个要发送的序列号
```

**执行流程**：

```python
async def speak(self, tts_text, display_text, actions, ...):
    # 1. 获取序列号
    current_sequence = self._sequence_counter
    self._sequence_counter += 1

    # 2. 启动发送器任务（如果未运行）
    if not self._sender_task or self._sender_task.done():
        self._sender_task = asyncio.create_task(
            self._process_payload_queue(websocket_send)
        )

    # 3. 创建TTS异步任务
    task = asyncio.create_task(
        self._process_tts(..., sequence_number=current_sequence)
    )
    self.task_list.append(task)
```

**有序交付机制**：

```python
async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
    buffered_payloads: Dict[int, Dict] = {}

    while True:
        payload, sequence_number = await self._payload_queue.get()
        buffered_payloads[sequence_number] = payload

        # 按顺序发送（即使后面序号的先完成，也会等待前面的）
        while self._next_sequence_to_send in buffered_payloads:
            next_payload = buffered_payloads.pop(self._next_sequence_to_send)
            await websocket_send(json.dumps(next_payload))
            self._next_sequence_to_send += 1
```

**优势**：
- 多句子并行生成，减少总等待时间
- 保证前端按正确顺序播放
- 避免因某一句生成慢而阻塞后续句子

### 3.3 音频负载准备

**文件**: `src/open_llm_vtuber/utils/stream_audio.py`

```python
def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,  # 20ms分块
    display_text: DisplayText = None,
    actions: Actions = None,
) -> dict[str, any]:
    # 1. 加载音频文件
    audio = AudioSegment.from_file(audio_path)

    # 2. 转换为WAV格式并base64编码
    audio_bytes = audio.export(format="wav").read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    # 3. 计算音量用于唇形同步
    volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    return {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,           # 归一化音量数组
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions,
    }
```

**关键参数**：
- `chunk_length_ms=20`: 20ms分块，用于Live2D唇形动画同步
- `volumes`: 每个块的RMS音量，前端用于驱动口型

---

## 4. 与Agent系统的集成

### 4.1 Agent输出类型

**文件**: `src/open_llm_vtuber/agent/output_types.py`

```python
@dataclass
class SentenceOutput(BaseOutput):
    display_text: DisplayText  # UI显示文本
    tts_text: str              # TTS朗读文本（可能不同，如翻译后）
    actions: Actions           # Live2D动作/表情

@dataclass
class AudioOutput(BaseOutput):
    audio_path: str            # 直接音频输出（如语音克隆）
    display_text: DisplayText
    transcript: str
    actions: Actions
```

### 4.2 对话处理流程

**文件**: `src/open_llm_vtuber/conversations/single_conversation.py`

```python
async def process_single_conversation(context, websocket_send, ...):
    # 每个对话创建独立的TTS管理器
    tts_manager = TTSTaskManager()

    # Agent以流式返回句子
    async for output_item in context.agent_engine.chat(batch_input):
        if isinstance(output_item, (SentenceOutput, AudioOutput)):
            response_part = await process_agent_output(
                output=output_item,
                tts_manager=tts_manager,
                ...
            )

    # 等待所有TTS任务完成
    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))
```

### 4.3 句子输出处理

**文件**: `src/open_llm_vtuber/conversations/conversation_utils.py`

```python
async def handle_sentence_output(output, ...):
    async for display_text, tts_text, actions in output:
        # 翻译支持
        if translate_engine:
            tts_text = translate_engine.translate(tts_text)

        # 提交到TTS管理器
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            ...
        )
```

---

## 5. WebSocket通信

### 5.1 主WebSocket端点

**文件**: `src/open_llm_vtuber/routes.py`

```python
@router.websocket("/client-ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_uid = str(uuid4())
    await ws_handler.handle_new_connection(websocket, client_uid)
    await ws_handler.handle_websocket_communication(websocket, client_uid)
```

### 5.2 专用TTS WebSocket端点

```python
@router.websocket("/tts-ws")
async def tts_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        text = data.get("text")

        # 简单分句处理
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        for sentence in sentences:
            audio_path = await tts_engine.async_generate_audio(text=sentence, ...)
            await websocket.send_json({
                "status": "partial",
                "audioPath": audio_path,
                "text": sentence,
            })
```

---

## 6. 各TTS引擎的流式支持情况

| 引擎 | 流式API | 实际实现 | 说明 |
|------|---------|----------|------|
| Edge TTS | ❌ | 同步保存文件 | `communicate.save_sync(file_name)` |
| OpenAI TTS | ✅ | 流式接收但保存文件 | `with_streaming_response.create()` + `stream_to_file()` |
| Cartesia | ✅ | 流式接收但保存文件 | `client.tts.bytes()` + 逐chunk写入文件 |
| Azure TTS | ✅ | 未使用流式 | 同步SDK |
| GPT-SoVITS | ✅ | 支持流式模式 | 配置`streaming_mode`参数 |
| SiliconFlow | ✅ | 支持流式 | 配置`stream`参数 |
| CosyVoice2 | ✅ | 支持流式 | 配置`stream`参数 |

**注意**：即使底层API支持流式，当前架构仍要求**完整文件生成后才能发送**。

---

## 7. 性能分析

### 7.1 延迟组成

```
总延迟 = LLM首字延迟 + 句子生成延迟 + TTS生成延迟 + 网络传输延迟
```

### 7.2 优化点

1. **并行TTS生成**：多句子同时生成，而非串行
2. **LLM流式输出**：Agent以句子为单位流式输出，TTS可以尽早开始
3. **线程池**：同步TTS引擎在线程池中运行，不阻塞事件循环

### 7.3 潜在瓶颈

1. **首句延迟**：必须等待第一句完整生成才能开始播放
2. **大文件传输**：长文本生成的音频文件较大，base64编码增加体积
3. **内存占用**：多个并行的完整音频文件缓存在内存中

---

## 8. 与真流式TTS的对比

### 8.1 当前方案（伪流式）

```
优点：
- 实现简单，兼容所有TTS引擎
- 易于实现唇形同步（有完整音频音量数据）
- 可靠性高，网络波动影响小

缺点：
- 首句延迟较高
- 内存占用随文本长度增加
- 不支持真正的"边说边播"
```

### 8.2 真流式方案（理论）

```
优点：
- 首音延迟极低（几百毫秒）
- 内存占用恒定
- 更接近实时对话体验

缺点：
- 需要TTS引擎原生支持流式
- 唇形同步更复杂（需基于音频流实时计算）
- 网络抖动影响播放连续性
- 实现复杂度高
```

---

## 9. 关键代码参考

### 9.1 TTS任务执行与清理

```python
# src/open_llm_vtuber/conversations/tts_manager.py:130-165
async def _process_tts(self, tts_text, ..., sequence_number: int) -> None:
    audio_file_path = None
    try:
        audio_file_path = await self._generate_audio(tts_engine, tts_text)
        payload = prepare_audio_payload(audio_path=audio_file_path, ...)
        await self._payload_queue.put((payload, sequence_number))
    except Exception as e:
        logger.error(f"Error preparing audio payload: {e}")
        # 错误时发送静音payload
        payload = prepare_audio_payload(audio_path=None, ...)
        await self._payload_queue.put((payload, sequence_number))
    finally:
        if audio_file_path:
            tts_engine.remove_file(audio_file_path)  # 及时清理缓存文件
```

### 9.2 音量计算用于唇形同步

```python
# src/open_llm_vtuber/utils/stream_audio.py:8-24
def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    max_volume = max(volumes)
    if max_volume == 0:
        raise ValueError("Audio is empty or all zero.")
    return [volume / max_volume for volume in volumes]  # 归一化到0-1
```

---

## 10. 总结

Open-LLM-VTuber 的 TTS 系统采用了一种务实的**伪流式架构**：

1. **设计哲学**：优先保证可靠性和唇形同步精度，而非极致的低延迟
2. **核心机制**：`TTSTaskManager` 通过序列号机制实现并行生成、有序交付
3. **扩展性**：工厂模式支持20+种TTS引擎，统一接口便于新增引擎
4. **优化空间**：如需更低延迟，可考虑实现真正的流式TTS（需要引擎支持和架构调整）

对于目标应用场景（Live2D虚拟角色对话），当前方案在延迟和质量之间取得了合理平衡。端到端延迟主要受限于LLM生成速度和TTS引擎本身，架构层面的优化空间有限。
