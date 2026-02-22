# Phase 0: 技术验证与准备

**目标**: 验证 LiveKit 技术栈可行性，搭建开发环境
**时长**: 1-2 天
**依赖**: 无

---

## TODO 0.1: LiveKit Server 部署

- [ ] 注册 LiveKit Cloud 账号（或本地 Docker 部署）
- [ ] 获取 `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- [ ] 测试连接：`lk debug` 命令验证服务器可达
- [ ] **存档点**: 保存环境变量到 `.env.livekit`

**参考命令**:
```bash
# Docker 本地部署
docker run --rm -p 7880:7880 \
  -e LIVEKIT_KEYS="devkey: secret" \
  livekit/livekit-server

# 测试连接
lk debug
```

---

## TODO 0.2: 运行 LiveKit 官方示例

- [ ] 安装依赖: `pip install livekit-agents livekit-plugins-silero`
- [ ] 下载并运行 `minimal_assistant.py`
- [ ] 浏览器访问 LiveKit Playground 测试连接
- [ ] **验证 AEC 效果**: 播放音乐同时说话，确认回声被消除
- [ ] **存档点**: 截图/录屏记录验证结果

**参考命令**:
```bash
# 下载示例
wget https://github.com/livekit/agents/raw/main/examples/voice-pipeline-agent/minimal_assistant.py

# 运行示例
python minimal_assistant.py
```

---

## TODO 0.3: 架构决策确认

- [ ] 确认前端框架选择（Vue / React / Vanilla）
- [ ] 确认 Tauri vs Electron（推荐 Tauri）
- [ ] 确认 LiveKit 部署方式（Cloud vs Self-hosted）
- [ ] **存档点**: 更新 `docs/ARCHITECTURE.md`

**决策记录模板**:
```markdown
## 决策记录
- 前端框架: Vue 3 / React / Vanilla
- 桌面打包: Tauri / Electron
- LiveKit 部署: Cloud / Self-hosted
- 理由: ...
```

---

## 完成标准

✅ 可以成功连接 LiveKit Room
✅ 验证 AEC 效果满足需求
✅ 架构决策已记录

---

**下一步**: [Phase 1: 后端 Plugin 包装](phase_1.md)
