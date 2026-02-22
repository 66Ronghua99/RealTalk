# Phase 1: 后端 Plugin 包装

**目标**: 将 RealTalk 现有组件包装为 LiveKit Agent Plugins
**时长**: 3-4 天
**依赖**: Phase 0

---

## TODO 1.1: 创建 Plugin 基础框架

- [ ] 创建 `src/realtalk/livekit_plugins/__init__.py`
- [ ] 创建 `src/realtalk/livekit_plugins/base.py`（Plugin 基类）
- [ ] 定义统一的 Plugin 接口规范
- [ ] **测试**: 验证 Plugin 可以被 LiveKit Agent 加载
- [ ] **存档点**: 提交基础框架代码

**关键代码模板**:
```python
# base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator

class BaseRealTalkPlugin(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass
```

---

## TODO 1.2: ASR Plugin 实现

- [ ] 创建 `src/realtalk/livekit_plugins/stt.py`
- [ ] 包装 `SherpaOnnxASR` 为 `RealTalkSTTPlugin`
- [ ] 实现 `speech_to_text()` 异步接口
- [ ] 处理音频格式转换（LiveKit AudioFrame → ASR 输入）
- [ ] **单元测试**: 验证 ASR Plugin 独立工作
- [ ] **集成测试**: 在 LiveKit Agent 中测试语音识别
- [ ] **存档点**: 提交 ASR Plugin 代码

**关键代码模板**:
```python
# stt.py
from livekit.agents import stt
from realtalk.perception.asr import SherpaOnnxASR

class RealTalkSTTPlugin(stt.STT):
    def __init__(self, asr: SherpaOnnxASR):
        self._asr = asr

    async def speech_to_text(self, audio_frame) -> str:
        # 转换音频格式
        # 调用 SherpaOnnxASR
        pass
```

---

## TODO 1.3: LLM Plugin 实现

- [ ] 创建 `src/realtalk/livekit_plugins/llm.py`
- [ ] 包装 `OpenRouterLLM` 为 `RealTalkLLMPlugin`
- [ ] 实现 `chat()` 异步接口
- [ ] 处理对话历史格式转换
- [ ] **单元测试**: 验证 LLM Plugin 独立工作
- [ ] **存档点**: 提交 LLM Plugin 代码

---

## TODO 1.4: TTS Plugin 实现

- [ ] 创建 `src/realtalk/livekit_plugins/tts.py`
- [ ] 包装 `MinimaxTTS` 为 `RealTalkTTSPlugin`
- [ ] 实现 `synthesize()` 异步接口
- [ ] 处理音频输出格式转换
- [ ] **单元测试**: 验证 TTS Plugin 独立工作
- [ ] **存档点**: 提交 TTS Plugin 代码

---

## TODO 1.5: Gatekeeper 回调集成

- [ ] 分析 LiveKit `before_llm_cb` 回调时机
- [ ] 创建 `src/realtalk/livekit/gatekeeper_callback.py`
- [ ] 将 `RuleBasedGatekeeper` 逻辑接入回调
- [ ] 实现 ACCUMULATE 延迟响应逻辑
- [ ] 实现 INTERRUPT 打断响应逻辑
- [ ] **测试**: 验证各种决策路径正常工作
- [ ] **存档点**: 提交 Gatekeeper 集成代码

---

## 完成标准

✅ ASR/LLM/TTS Plugins 可独立运行
✅ Gatekeeper 集成到 LiveKit 回调
✅ 所有 Plugins 通过单元测试

---

**前置**: [Phase 0: 技术验证](phase_0.md)
**下一步**: [Phase 2: Agent 主程序](phase_2.md)
