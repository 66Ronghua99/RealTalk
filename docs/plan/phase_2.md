# Phase 2: LiveKit Agent 主程序

**目标**: 创建完整的 LiveKit Agent Worker
**时长**: 2-3 天
**依赖**: Phase 1

---

## TODO 2.1: Agent 入口程序

- [ ] 创建 `src/realtalk/livekit_agent.py`
- [ ] 实现 `entrypoint()` 函数
- [ ] 集成所有 Plugins（ASR/LLM/TTS/VAD）
- [ ] 实现 Room 连接和事件处理
- [ ] **测试**: Agent 可以成功连接 LiveKit Room
- [ ] **存档点**: 提交 Agent 入口代码

**关键代码模板**:
```python
# livekit_agent.py
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import VoicePipelineAgent

class RealTalkAgent:
    async def entrypoint(self, ctx: JobContext):
        await ctx.connect()

        agent = VoicePipelineAgent(
            vad=silero.VAD(),
            stt=RealTalkSTTPlugin(self.asr),
            llm=RealTalkLLMPlugin(self.llm),
            tts=RealTalkTTSPlugin(self.tts),
            before_llm_cb=self._gatekeeper_check,
        )

        agent.start(ctx.room)

if __name__ == "__main__":
    agent = RealTalkAgent()
    cli.run_app(WorkerOptions(entrypoint_fnc=agent.entrypoint))
```

---

## TODO 2.2: 对话状态管理

- [ ] 分析 `ConversationManager` 与 LiveKit 的集成点
- [ ] 实现对话历史同步
- [ ] 处理多轮对话上下文
- [ ] **测试**: 验证对话历史正确维护
- [ ] **存档点**: 提交状态管理代码

---

## TODO 2.3: 打断检测集成

- [ ] 分析 LiveKit 打断机制（`interrupt()` 方法）
- [ ] 集成 RealTalk 的打断逻辑
- [ ] 实现用户语音触发打断
- [ ] 实现按钮触发打断（保留作为 fallback）
- [ ] **测试**: 验证打断功能正常工作
- [ ] **存档点**: 提交打断检测代码

---

## TODO 2.4: 配置管理

- [ ] 创建 `src/realtalk/livekit_config.py`
- [ ] 集成 LiveKit 配置与现有 `config.py`
- [ ] 支持环境变量加载
- [ ] 创建 `.env.example` 模板
- [ ] **测试**: 验证配置加载正确
- [ ] **存档点**: 提交配置管理代码

**配置模板**:
```bash
# .env.example
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

---

## 完成标准

✅ Agent 可以独立运行并处理对话
✅ 打断检测正常工作
✅ 配置管理完善

---

**前置**: [Phase 1: 后端 Plugin](phase_1.md)
**下一步**: [Phase 3: 前端开发](phase_3.md)
