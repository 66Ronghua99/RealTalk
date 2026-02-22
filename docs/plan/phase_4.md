# Phase 4: 桌面客户端打包

**目标**: 用 Tauri 打包为 Mac/Windows 桌面应用
**时长**: 2-3 天
**依赖**: Phase 3

---

## TODO 4.1: Tauri 项目初始化

- [ ] 在前端项目初始化 Tauri: `npm run tauri init`
- [ ] 配置 `tauri.conf.json`
- [ ] 配置应用图标、名称、版本
- [ ] **测试**: 验证 `npm run tauri dev` 可以启动
- [ ] **存档点**: 提交 Tauri 配置

**初始化命令**:
```bash
cd frontend
npm run tauri init

# 配置选项
# - app name: realtalk
# - window title: RealTalk
# - dev path: http://localhost:5173
# - dist dir: ../dist
```

---

## TODO 4.2: 原生功能集成

- [ ] 实现系统托盘图标
- [ ] 实现全局快捷键（如 Cmd+Shift+Space 唤醒）
- [ ] 实现菜单栏（Mac）/ 系统菜单（Windows）
- [ ] **测试**: 验证原生功能正常工作
- [ ] **存档点**: 提交原生功能代码

**Tauri 配置示例**:
```json
{
  "tauri": {
    "systemTray": {
      "iconPath": "icons/icon.png",
      "iconAsTemplate": true
    },
    "globalShortcuts": [
      {
        "shortcut": "CmdOrControl+Shift+Space",
        "action": "toggle_window"
      }
    ]
  }
}
```

---

## TODO 4.3: 打包与签名

- [ ] Mac 打包: `npm run tauri build --target universal-apple-darwin`
- [ ] Windows 打包: `npm run tauri build --target x86_64-pc-windows-msvc`
- [ ] 配置自动更新（可选）
- [ ] **测试**: 验证安装包可以正常安装运行
- [ ] **存档点**: 保存打包脚本和配置

**输出文件**:
```
src-tauri/target/release/bundle/
├── macos/RealTalk.app
├── dmg/RealTalk_0.1.0_x64.dmg
├── msi/RealTalk_0.1.0_x64_en-US.msi
└── exe/RealTalk_0.1.0_x64-setup.exe
```

---

## 完成标准

✅ Mac 安装包可以正常安装运行
✅ Windows 安装包可以正常安装运行
✅ 系统托盘和快捷键正常工作

**里程碑 M3**: 桌面客户端可用

---

**前置**: [Phase 3: 前端开发](phase_3.md)
**下一步**: [Phase 5: 废弃旧代码](phase_5.md)
