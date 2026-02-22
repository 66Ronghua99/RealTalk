# Phase 3: 前端开发（浏览器）

**目标**: 创建浏览器端用户界面
**时长**: 5-7 天
**依赖**: Phase 0（可与 Phase 1/2 并行）

---

## TODO 3.1: 前端项目初始化

- [ ] 创建 `frontend/` 目录
- [ ] 初始化项目（推荐 Vite + Vue3 或 React）
- [ ] 安装 LiveKit Client SDK: `npm install livekit-client`
- [ ] 配置开发代理（解决 CORS）
- [ ] **测试**: 验证项目可以正常启动
- [ ] **存档点**: 提交前端初始代码

**初始化命令**:
```bash
npm create vue@latest frontend
# 或
npm create vite@latest frontend -- --template react-ts
```

---

## TODO 3.2: LiveKit 连接管理

- [ ] 创建 `src/services/livekit.ts` 连接服务
- [ ] 实现 Token 获取（从后端或生成）
- [ ] 实现 Room 连接/断开
- [ ] 实现连接状态显示
- [ ] **测试**: 验证可以成功连接 LiveKit Room
- [ ] **存档点**: 提交连接管理代码

**关键代码模板**:
```typescript
// services/livekit.ts
import { Room, RoomEvent } from 'livekit-client';

export class LiveKitService {
  private room: Room | null = null;

  async connect(url: string, token: string): Promise<void> {
    this.room = new Room({
      adaptiveStream: true,
      dynacast: true,
    });

    await this.room.connect(url, token);
  }

  async disconnect(): Promise<void> {
    await this.room?.disconnect();
  }
}
```

---

## TODO 3.3: 音频采集与播放

- [ ] 实现麦克风权限获取
- [ ] 实现音频 track 发布
- [ ] 实现 Agent 音频订阅和播放
- [ ] 实现音量可视化
- [ ] **测试**: 验证音频双向流通
- [ ] **存档点**: 提交音频处理代码

---

## TODO 3.4: UI 组件开发

- [ ] 创建状态显示组件（IDLE/LISTENING/SPEAKING/PROCESSING）
- [ ] 创建对话历史组件
- [ ] 创建打断按钮组件
- [ ] 创建设置面板（选择设备、调整参数）
- [ ] **测试**: 验证所有 UI 组件正常工作
- [ ] **存档点**: 提交 UI 组件代码

**组件列表**:
- `StatusIndicator.vue` - 状态指示器
- `ChatHistory.vue` - 对话历史
- `InterruptButton.vue` - 打断按钮
- `SettingsPanel.vue` - 设置面板

---

## TODO 3.5: 状态同步

- [ ] 实现前端状态与 LiveKit Room 状态同步
- [ ] 实现对话历史显示
- [ ] 实现实时转写显示
- [ ] **测试**: 验证状态同步正确
- [ ] **存档点**: 提交状态同步代码

---

## TODO 3.6: 响应式适配

- [ ] 移动端适配
- [ ] 不同屏幕尺寸适配
- [ ] 暗黑模式支持（可选）
- [ ] **测试**: 验证多设备兼容
- [ ] **存档点**: 提交响应式代码

---

## 完成标准

✅ 前端可以连接 LiveKit Room
✅ 音频双向流通正常
✅ UI 组件完整，状态显示正确
✅ 支持响应式布局

**里程碑 M2**: 浏览器版本可用

---

**前置**: [Phase 0: 技术验证](phase_0.md)
**下一步**: [Phase 4: 桌面打包](phase_4.md)
