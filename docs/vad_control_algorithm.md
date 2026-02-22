# VAD 控制算法详解

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    VAD 算法流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  音频输入 (16kHz)                                            │
│       ↓                                                     │
│  ┌─────────────┐    512 样本 (32ms)    ┌─────────────┐     │
│  │  分帧处理    │ ────────────────────▶ │ Silero VAD  │     │
│  │  (滑动窗口)  │                       │  神经网络    │     │
│  └─────────────┘                       └──────┬──────┘     │
│                                               ↓             │
│                                         speech_prob         │
│                                               ↓             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              状态机 (StateMachine)                   │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐         │   │
│  │  │  IDLE   │───▶│ ACTIVE  │───▶│ INACTIVE│         │   │
│  │  │ (等待)   │    │ (说话中) │    │ (检测结束)│        │   │
│  │  └─────────┘    └────┬────┘    └────┬────┘         │   │
│  │                      │              │               │   │
│  │                      └──────────────┘               │   │
│  │                              (循环检测新语音)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 2. 语音概率计算 (Silero VAD 神经网络)

**核心代码** (`src/open_llm_vtuber/vad/silero.py:61-62`):
```python
with torch.no_grad():
    speech_prob = self.model(chunk, self.config.target_sr).item()
```

### Silero VAD 内部原理

Silero VAD 是一个预训练的**深度神经网络**，其架构特点：

1. **输入**: 32ms 音频帧 (512 样本 @ 16kHz)
2. **特征提取**: 使用 **STFT** (短时傅里叶变换) 提取频谱特征
3. **网络结构**:
   - 基于 **Temporal Convolutional Network (TCN)**
   - 或轻量级 **RNN** 变体 (取决于模型版本)
4. **输出**: 0-1 之间的概率值

### 概率含义

| 概率值 | 含义 |
|--------|------|
| 0.0 | 确定是静音/噪音 |
| 0.5 | 不确定 |
| 1.0 | 确定是语音 |

**默认阈值**: `prob_threshold = 0.4` (可配置)

## 3. 状态机算法详解

### 三个状态

```python
class State(Enum):
    IDLE = 1      # 空闲，等待语音开始
    ACTIVE = 2    # 检测到语音，持续收集
    INACTIVE = 3  # 语音结束，准备输出
```

### 状态转换逻辑

```
                    ┌──────────────┐
         (hit_count  │              │
          < 3)       │    IDLE      │◀────────────────┐
    ┌─────────────── │   (等待语音)  │                 │
    │                │              │                 │
    │                └──────┬───────┘                 │
    │                       │ prob < 0.4 or db < 60   │
    │                       │ (连续24次)               │
    │    prob >= 0.4        ▼                         │
    │    db >= 60      ┌──────────────┐    miss_count  │
    │    (连续3次)     │   INACTIVE    │    >= 24      │
    └──────────────▶   │   (语音结束)   │───────────────┘
                       └──────┬───────┘
                              │ prob >= 0.4
                              │ db >= 60
                              │ (连续3次)
                              ▼
                       ┌──────────────┐
                       │    ACTIVE    │
                       │   (语音收集中) │
                       └──────────────┘
```

### 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `prob_threshold` | 0.4 | 语音概率阈值 |
| `db_threshold` | 60 | 音量阈值 (dB) |
| `required_hits` | 3 | 确认语音需要连续 3 帧 (约 96ms) |
| `required_misses` | 24 | 确认结束需要连续 24 帧 (约 768ms) |
| `smoothing_window` | 5 | 平滑窗口大小 |

## 4. 平滑处理算法

**代码** (`src/open_llm_vtuber/vad/silero.py:120-125`):
```python
def get_smoothed_values(self, prob, db):
    self.prob_window.append(prob)
    self.db_window.append(db)
    smoothed_prob = np.mean(self.prob_window)
    smoothed_db = np.mean(self.db_window)
    return smoothed_prob, smoothed_db
```

使用 **滑动窗口平均** 减少抖动：
- 窗口大小: 5 帧 (约 160ms)
- 对新值和最近 4 个历史值取平均

## 5. 音量计算 (dB)

**代码** (`src/open_llm_vtuber/vad/silero.py:106-108`):
```python
@classmethod
def calculate_db(cls, audio_data: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(audio_data)))  # 均方根
    return 20 * np.log10(rms + 1e-7) if rms > 0 else -np.inf
```

**公式**:
```
dB = 20 * log10(RMS)
其中 RMS = sqrt(mean(x²))
```

## 6. 预缓冲区机制

**代码** (`src/open_llm_vtuber/vad/silero.py:103`, `136`):
```python
self.pre_buffer = deque(maxlen=20)  # 约 640ms

# 在 IDLE 状态时缓存音频
if self.state == State.IDLE:
    self.pre_buffer.append(chunk_bytes)
```

**作用**: 避免漏掉语音开头的部分，将开始检测前的 20 帧音频也包含在输出中。

## 7. 信号输出

| 信号 | 含义 | 触发条件 |
|------|------|---------|
| `b"<|PAUSE|>"` | 检测到语音开始 | IDLE → ACTIVE 转换 |
| `b"<|RESUME|>"` | 语音结束，返回空闲 | INACTIVE → IDLE 转换 |
| `audio_bytes` | 收集的语音数据 | 语音结束后输出 |

## 8. 算法伪代码

```python
for each 32ms audio_chunk:
    # 1. 神经网络推理
    prob = silero_vad_model(audio_chunk)

    # 2. 计算音量
    db = 20 * log10(RMS(audio_chunk))

    # 3. 平滑处理
    smoothed_prob = mean(last_5_probs)
    smoothed_db = mean(last_5_dbs)

    # 4. 状态机处理
    if state == IDLE:
        if smoothed_prob >= 0.4 and smoothed_db >= 60:
            hit_count += 1
            if hit_count >= 3:  # 连续3帧确认
                state = ACTIVE
                yield "<|PAUSE|>"
        else:
            hit_count = 0
            pre_buffer.append(audio_chunk)  # 缓存预音频

    elif state == ACTIVE:
        collect_audio(audio_chunk)
        if smoothed_prob < 0.4 or smoothed_db < 60:
            miss_count += 1
            if miss_count >= 24:  # 连续24帧静音
                state = INACTIVE
        else:
            miss_count = 0

    elif state == INACTIVE:
        if smoothed_prob >= 0.4 and smoothed_db >= 60:
            # 新的语音开始
            state = ACTIVE
        elif miss_count >= 24:
            state = IDLE
            yield "<|RESUME|>"
            yield collected_audio  # 输出完整语音
```

## 9. 与打断机制的结合

在 `websocket_handler.py` 中：

```python
async def _handle_raw_audio_data(self, websocket, client_uid, data):
    chunk = data.get("audio", [])
    for audio_bytes in context.vad_engine.detect_speech(chunk):
        if audio_bytes == b"<|PAUSE|>":
            # 检测到语音，发送打断信号
            await websocket.send_text(
                json.dumps({"type": "control", "text": "interrupt"})
            )
        elif len(audio_bytes) > 1024:
            # 检测到完整语音片段，累积到缓冲区
            self.received_data_buffers[client_uid] = np.append(...)
```

## 10. 配置示例

```yaml
vad_config:
  silero_vad:
    orig_sr: 16000           # 原始采样率
    target_sr: 16000         # 目标采样率
    prob_threshold: 0.4      # 语音概率阈值
    db_threshold: 60         # 音量阈值 (dB)
    required_hits: 3         # 确认语音需要的连续帧数
    required_misses: 24      # 确认结束需要的连续帧数
    smoothing_window: 5      # 平滑窗口大小
```

## 总结

这个 VAD 算法是**双重阈值 + 状态机**的设计：

1. **神经网络**提供语音概率（基于频谱特征学习）
2. **音量阈值**过滤环境噪音
3. **状态机**通过 hit/miss 计数器实现**迟滞效果**，避免频繁切换
4. **平滑窗口**减少短时抖动
5. **预缓冲区**确保语音开头不被截断
