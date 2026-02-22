# Phase 5: 废弃旧代码

**目标**: 清理不再需要的代码
**时长**: 1 天
**依赖**: Phase 2, Phase 4

---

## TODO 5.1: CLI 代码归档

- [ ] 创建 `archive/cli_backup/` 目录
- [ ] 移动 `src/realtalk/cli.py` 到归档目录
- [ ] 移动 `src/realtalk/transport/webrtc.py` 到归档目录
- [ ] 更新 `pyproject.toml` 移除 CLI 命令
- [ ] **存档点**: 提交归档操作

**归档结构**:
```
archive/
├── cli_backup/
│   ├── cli.py
│   └── webrtc.py
└── README.md  # 说明归档原因
```

---

## TODO 5.2: 旧服务器代码清理

- [ ] 分析 `src/realtalk/web/server.py` 是否完全废弃
- [ ] 如废弃，移动到归档目录
- [ ] 保留可能复用的工具函数
- [ ] **存档点**: 提交清理代码

---

## TODO 5.3: 文档更新

- [ ] 更新 `README.md` 说明新架构
- [ ] 更新 `Progress.md` 标记旧阶段完成
- [ ] 创建 `docs/MIGRATION.md` 说明迁移过程
- [ ] **存档点**: 提交文档更新

---

## 完成标准

✅ 旧代码已归档，不污染主代码库
✅ 文档已更新，说明新架构
✅ `pyproject.toml` 已清理

---

**前置**: [Phase 2: Agent 主程序](phase_2.md), [Phase 4: 桌面打包](phase_4.md)
**下一步**: [Phase 6: 集成测试](phase_6.md)
