# Open-LLM-VTuber Streaming 架构深度分析

## 概述

本文档全面分析 Open-LLM-VTuber 项目中 **Streaming（流式处理）** 的设计与实现，涵盖从 LLM 文本生成到 TTS 音频输出的完整数据流。

**核心设计理念**：采用**多级流式流水线**架构，实现"边生成、边处理、边播放"的实时对话体验。

---

## 1. 整体架构概览

### 1.1 Streaming 流程全景图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Streaming Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User Input                                                                      │
│      ↓                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Stage 1: LLM Streaming                               │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │   │
│  │  │   Claude/    │    │   OpenAI/    │    │   Llama.cpp  │               │   │
│  │  │   AsyncStream│    │   AsyncStream│    │   create_    │               │   │
│  │  │              │    │              │    │   chat_      │               │   │
│  │  │  AsyncIterator│   │  AsyncIterator│   │  completion  │               │   │
│  │  │  [token...]  │    │  [token...]  │    │  (stream=True)│              │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘               │   │
│  │                              ↓                                          │   │
│  │                    AsyncIterator[str] 或 Dict                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│      ↓                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                   Stage 2: Agent Pipeline (Decorators)                   │   │
│  │                                                                          │   │
│  │   @tts_filter ← @display_processor ← @actions_extractor ← @sentence_   │   │
│  │   _divider                                                                 │   │
│  │                                                                          │   │
│  │   Token Stream → SentenceWithTags → (Sentence, Actions) →              │   │
│  │   (Sentence, DisplayText, Actions) → SentenceOutput                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│      ↓                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                   Stage 3: TTS Parallel Processing                       │   │
│  │                                                                          │   │
│  │   TTSTaskManager: 并行生成 + 有序交付                                     │   │
│  │                                                                          │   │
│  │   SentenceOutput ─┬─→ TTS Task 0 (seq=0) ─┐                            │   │
│  │                   ├─→ TTS Task 1 (seq=1) ─┼─→ Ordered Queue ─→ WebSocket│   │
│  │                   └─→ TTS Task 2 (seq=2) ─┘                            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│      ↓                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                   Stage 4: WebSocket Streaming                           │   │
│  │                                                                          │   │
│  │   Payload: {audio: base64, volumes: [...], display_text: {...},         │   │
│  │            actions: {...}}                                               │   │
│  │                                                                          │   │
│  │   JSON → WebSocket → Frontend → Base64 Decode → AudioBuffer → Play      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Streaming 的三大层次

| 层次 | 类型 | 关键组件 | 说明 |
|------|------|----------|------|
| **LLM Streaming** | 真流式 | `StatelessLLMInterface` | Token/Chunk 级实时生成 |
| **Agent Pipeline** | 流式转换 | `sentence_divider` 等装饰器 | Token → 句子的流式聚合 |
| **TTS Processing** | 伪流式 | `TTSTaskManager` | 并行生成 + 有序交付 |

---

## 2. LLM Streaming 层

### 2.1 统一接口设计

**文件**: `src/open_llm_vtuber/agent/stateless_llm/stateless_llm_interface.py`

```python
class StatelessLLMInterface(ABC):
    @abstractmethod
    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str = None, tools: List[Dict] = None
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Return an async iterator yielding tokens or event dicts"""
        raise NotImplementedError
```

**关键设计**:
- 所有LLM实现返回 `AsyncIterator`，支持真正的异步流式
- 统一事件格式: `{"type": "text_delta", "text": "..."}` 或纯字符串
- 支持 Tool Calling 的流式处理

### 2.2 各LLM实现对比

| LLM Provider | 流式实现 | 特点 |
|--------------|----------|------|
| **Claude** | `async with client.messages.stream() as stream` | 原生异步，支持 `text_delta`, `tool_use_start`, `message_stop` 等事件 |
| **OpenAI Compatible** | `client.chat.completions.create(stream=True)` | 使用 `AsyncStream[ChatCompletionChunk]` |
| **Llama.cpp** | `create_chat_completion(stream=True)` + `asyncio.to_thread` | 同步API在线程池中运行，yield生成器结果 |
| **Template-based** | `requests.post(stream=True)` | 原始HTTP流，逐行解析 |

### 2.3 Claude 流式实现示例

**文件**: `src/open_llm_vtuber/agent/stateless_llm/claude_llm.py`

```python
async def chat_completion(self, messages, system=None, tools=None):
    async with self.client.messages.stream(
        model=self.model,
        max_tokens=self.max_tokens,
        system=system if system else self._system,
        messages=messages,
        tools=tools,
    ) as stream:
        async for event in stream:
            if event.type == "text":
                yield {"type": "text_delta", "text": event.text}
            elif event.type == "tool_use":
                yield {"type": "tool_use_start", "data": {...}}
            elif event.type == "message_stop":
                yield {"type": "message_stop"}
```

### 2.4 OpenAI 流式实现示例

**文件**: `src/open_llm_vtuber/agent/stateless_llm/openai_compatible_llm.py`

```python
async def chat_completion(self, messages, system=None, tools=None):
    stream = await self.client.chat.completions.create(
        model=self.model,
        messages=messages,
        stream=True,  # 启用流式
        tools=tools,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content  # 直接yield字符串
        elif chunk.choices[0].delta.tool_calls:
            yield [{"id": ..., "name": ..., "arguments": ...}]
```

---

## 3. Agent Pipeline 层

### 3.1 装饰器流水线架构

**文件**: `src/open_llm_vtuber/agent/agents/basic_memory_agent.py:581-662`

```python
def _chat_function_factory(self):
    @tts_filter(self._tts_preprocessor_config)           # Stage 4: TTS过滤
    @display_processor()                                  # Stage 3: 显示处理
    @actions_extractor(self._live2d_model)                # Stage 2: 动作提取
    @sentence_divider(                                   # Stage 1: 分句
        faster_first_response=self._faster_first_response,
        segment_method=self._segment_method,
    )
    async def chat_with_memory(input_data: BatchInput):
        # 获取LLM流式输出
        token_stream = self._llm.chat_completion(messages, self._system)
        async for event in token_stream:
            yield event  # 装饰器会自动转换

    return chat_with_memory
```

**数据流转换**:

```
LLM Output (str/dict)
    ↓
@sentence_divider
    ↓
SentenceWithTags (带标签的句子)
    ↓
@actions_extractor
    ↓
Tuple[SentenceWithTags, Actions] (句子+Live2D动作)
    ↓
@display_processor
    ↓
Tuple[SentenceWithTags, DisplayText, Actions] (句子+显示文本+动作)
    ↓
@tts_filter
    ↓
SentenceOutput (最终输出，包含display_text, tts_text, actions)
```

### 3.2 SentenceDivider: Token → 句子

**文件**: `src/open_llm_vtuber/utils/sentence_divider.py`

```python
class SentenceDivider:
    def __init__(self, faster_first_response=True, segment_method="pysbd"):
        self.buffer = ""              # Token缓冲区
        self.sentence_enders = {'.', '!', '?', '。', '！', '？'}
        self.first_sentence_sent = False
        self.faster_first_response = faster_first_response

    async def process_stream(self, stream):
        async for token in stream:
            self.buffer += token

            # 快速首句：第一句按逗号分割
            if not self.first_sentence_sent and self.faster_first_response:
                if ',' in self.buffer or '，' in self.buffer:
                    parts = re.split(r'([,，])', self.buffer, maxsplit=1)
                    if len(parts) >= 2:
                        yield SentenceWithTags(text=parts[0] + parts[1])
                        self.buffer = "".join(parts[2:])
                        self.first_sentence_sent = True
                        continue

            # 标准分句：按句子结束符
            if any(e in self.buffer for e in self.sentence_enders):
                sentences = self._segment(self.buffer)
                for sent in sentences[:-1]:  # 除了最后一个（可能不完整）
                    yield SentenceWithTags(text=sent)
                self.buffer = sentences[-1]

        # 刷新剩余buffer
        if self.buffer:
            yield SentenceWithTags(text=self.buffer)
```

**优化点**: `faster_first_response` 让第一句在逗号处断开，更快开始TTS。

### 3.3 装饰器实现细节

**文件**: `src/open_llm_vtuber/agent/transformers.py`

所有装饰器遵循相同模式：

```python
def sentence_divider(...):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            stream = func(*args, **kwargs)  # 获取上游流
            async for item in divider.process_stream(stream):
                yield item  # 转换后yield
        return wrapper
    return decorator
```

---

## 4. TTS Processing 层

### 4.1 伪流式设计

**核心问题**: TTS引擎通常需要完整文本才能生成音频（非流式API）。

**解决方案**: `TTSTaskManager` 实现**并行生成 + 有序交付**。

**文件**: `src/open_llm_vtuber/conversations/tts_manager.py`

```python
class TTSTaskManager:
    def __init__(self):
        self.task_list: List[asyncio.Task] = []
        self._payload_queue: asyncio.Queue[Tuple[Dict, int]] = asyncio.Queue()
        self._sequence_counter = 0           # 分配序列号
        self._next_sequence_to_send = 0      # 下一个要发送的序号

    async def speak(self, tts_text, display_text, actions, ...):
        current_sequence = self._sequence_counter
        self._sequence_counter += 1

        # 启动发送器（如果未运行）
        if not self._sender_task:
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )

        # 创建TTS任务（并行执行）
        task = asyncio.create_task(
            self._process_tts(..., sequence_number=current_sequence)
        )
        self.task_list.append(task)
```

### 4.2 有序交付机制

```python
async def _process_payload_queue(self, websocket_send):
    buffered_payloads: Dict[int, Dict] = {}  # 序号 → payload

    while True:
        payload, seq = await self._payload_queue.get()
        buffered_payloads[seq] = payload

        # 按序发送（即使后面的先完成也等待）
        while self._next_sequence_to_send in buffered_payloads:
            await websocket_send(
                json.dumps(buffered_payloads.pop(self._next_sequence_to_send))
            )
            self._next_sequence_to_send += 1
```

### 4.3 与真流式TTS的对比

| 特性 | 当前方案（伪流式） | 真流式TTS（理论） |
|------|-------------------|-------------------|
| **首音延迟** | 高（等整句生成） | 低（几百毫秒） |
| **内存占用** | 随文本长度增加 | 恒定 |
| **实现复杂度** | 低 | 高 |
| **兼容性** | 支持所有TTS引擎 | 需引擎原生支持 |
| **唇形同步** | 精确（有完整音频） | 复杂（需实时计算） |

---

## 5. WebSocket Streaming 层

### 5.1 消息类型设计

**文件**: `src/open_llm_vtuber/websocket_handler.py:32-45`

```python
class MessageType(Enum):
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONTROL = ["interrupt-signal", "audio-play-start"]
    DATA = ["mic-audio-data"]
```

### 5.2 流式通信协议

```
Client → Server                    Server → Client
────────────────────────────────────────────────────────
mic-audio-data  ──────────────▶
   (音频流)                         full-text
                                    (状态更新)
mic-audio-end   ──────────────▶
   (触发对话)                       user-input-transcription
                                    (ASR结果显示)
                                   ──────────────────────────
                                    conversation-chain-start
                                      ↓
                                    audio  ←  流式发送多个
                                      ↓
                                    backend-synth-complete
                                      ↓
                                    conversation-chain-end
```

### 5.3 Audio Payload 结构

**文件**: `src/open_llm_vtuber/utils/stream_audio.py:27-82`

```python
def prepare_audio_payload(audio_path, chunk_length_ms=20, ...):
    audio = AudioSegment.from_file(audio_path)
    audio_bytes = audio.export(format="wav").read()

    return {
        "type": "audio",
        "audio": base64.b64encode(audio_bytes).decode(),  # Base64编码
        "volumes": _get_volume_by_chunks(audio, chunk_length_ms),  # 唇形同步
        "slice_length": chunk_length_ms,  # 20ms分块
        "display_text": {"text": ..., "name": ..., "avatar": ...},
        "actions": {"expressions": [...], "pictures": [...]},
    }
```

---

## 6. 关键设计决策

### 6.1 为什么选择伪流式TTS？

1. **兼容性优先**: 支持20+种TTS引擎，大多数不支持真流式
2. **唇形同步**: Live2D需要精确的音量数据，真流式难以实时计算
3. **可靠性**: 文件级生成更稳定，网络抖动不影响播放
4. **复杂度权衡**: 真流式需要重构TTS引擎接口和前端播放逻辑

### 6.2 Agent Pipeline 的价值

```
传统方式:
LLM输出全文 → 分句 → TTS → 播放
     (用户等待所有TTS完成)

Pipeline方式:
LLM流式输出 ─┬─→ 句子1 → TTS1 ─┐
             ├─→ 句子2 → TTS2 ─┼─→ 有序播放（用户更快听到第一句）
             └─→ 句子3 → TTS3 ─┘
```

### 6.3 中断处理

**文件**: `src/open_llm_vtuber/websocket_handler.py:369-392`

```python
async def _handle_interrupt(self, websocket, client_uid, data):
    heard_response = data.get("text", "")  # 用户听到了多少

    # 取消当前对话任务
    task = self.current_conversation_tasks.get(client_uid)
    if task and not task.done():
        task.cancel()

    # 通知Agent被中断
    context.agent_engine.handle_interrupt(heard_response)
```

---

## 7. 性能优化点

### 7.1 已实现的优化

| 优化点 | 实现位置 | 效果 |
|--------|----------|------|
| **并行TTS** | `TTSTaskManager` | 多句子同时生成 |
| **快速首句** | `sentence_divider` | 首句按逗号分割，减少等待 |
| **异步I/O** | 全链路 | 非阻塞，支持高并发 |
| **线程池** | `async_generate_audio` | 同步TTS引擎不阻塞事件循环 |

### 7.2 潜在优化方向

1. **真流式TTS**: 对于支持流式的引擎（OpenAI, Cartesia），直接流传输chunk
2. **预测性TTS**: 预生成常见回复的音频缓存
3. **WebSocket压缩**: 对音频payload进行压缩传输
4. **前端预缓冲**: 提前加载下一句音频

---

## 8. 代码关键路径

### 8.1 LLM → TTS 完整调用链

```
1. websocket_handler.py:513
   _handle_conversation_trigger()
       ↓
2. conversations/conversation_handler.py
   handle_conversation_trigger()
       ↓
3. conversations/single_conversation.py
   process_single_conversation()
       ↓
4. agent/agents/basic_memory_agent.py:664
   chat() → _chat_function_factory() 生成的装饰器链
       ↓
5. stateless_llm/xxx_llm.py
   chat_completion() 返回 AsyncIterator
       ↓
6. agent/transformers.py
   sentence_divider → actions_extractor → display_processor → tts_filter
       ↓
7. conversations/conversation_utils.py
   process_agent_output() → handle_sentence_output()
       ↓
8. conversations/tts_manager.py
   TTSTaskManager.speak() → 并行生成 → 有序队列
       ↓
9. utils/stream_audio.py
   prepare_audio_payload()
       ↓
10. WebSocket.send_text()
```

### 8.2 关键文件索引

| 文件 | 职责 |
|------|------|
| `agent/stateless_llm/*.py` | LLM流式生成 |
| `agent/transformers.py` | 流式数据转换（装饰器） |
| `agent/agents/basic_memory_agent.py` | Agent主逻辑，Pipeline组装 |
| `conversations/tts_manager.py` | TTS并行管理 |
| `conversations/single_conversation.py` | 单聊流程编排 |
| `utils/stream_audio.py` | 音频处理与打包 |
| `websocket_handler.py` | WebSocket消息路由 |

---

## 9. 总结

Open-LLM-VTuber 的 Streaming 架构是一个**分层渐进式**的设计：

1. **LLM层**: 真正的Token级流式，最大化利用大模型的生成速度
2. **Agent层**: 流式转换，将Token聚合成句子，同时进行情感提取和文本处理
3. **TTS层**: 伪流式（并行生成+有序交付），平衡延迟与兼容性
4. **传输层**: JSON over WebSocket，简单通用

**设计哲学**: 在"实时性"和"可靠性"之间找到平衡点，优先保证对话的自然流畅，而非极致的低延迟。

**适用场景**: Live2D虚拟角色对话，需要精确的唇形同步和稳定的音频播放。
