# RitonCode 智能编码代理系统

这是一个基于大语言模型的智能编码代理系统，能够理解项目结构、执行代码任务、管理待办事项、持久化任务，并支持技能扩展。

## 核心特性

### 1. **智能代理循环 (Agent Loop)**
- 自动化的推理和执行循环
- 支持消息压缩以优化长对话上下文

### 2. **强大的工具集 (Tools)**
- **文件操作**: 读取、写入、编辑文件
- **命令执行**: 安全的bash命令执行（带危险命令过滤）
- **待办管理**: 跟踪任务进度，自动提醒更新计划
- **任务管理（新增）**: 支持创建、查询、列出、更新持久化任务及依赖关系
- **子代理支持**: 在隔离环境中运行子任务
- **技能系统**: 可扩展的技能框架，允许用户添加自定义技能
- **定时任务 (Cron)**: 支持定时触发任务，自动执行预设操作

### 3. **技能系统 (Skills)**
- 可扩展的技能框架，允许用户添加自定义技能
- 技能通过YAML前端元数据定义，Markdown主体内容实现

### 4. **安全设计**
- 文件路径沙箱：确保所有文件操作都在工作目录内
- 危险命令过滤：阻止`rm -rf/`、`dd`、`mkfs`等危险命令
- 命令超时限制：30秒超时防止挂起
- 输出长度限制：防止过长输出影响性能

### 5. **权限管理系统**
- **多级权限控制**：支持allow/deny/ask三种权限行为
- **Bash安全验证**：实时检测并阻止危险的shell命令
- **灵活的运行模式**：
  - `default`：标准模式，根据规则和用户确认决定权限
  - `plan`：规划模式，只允许读取操作，禁止所有写入工具
  - `auto`：自动批准工具调用；Bash 命令通过 Bubblewrap 沙箱执行
- **智能拒绝机制**：连续拒绝达到阈值时自动建议切换到更合适的模式
- **用户友好的交互**：支持一次性确认(y)、永久允许(always)或拒绝(n)

## 项目结构

```
.
├── config.yaml               # AI模型配置文件（自行配置）
├── src/                      # 核心源代码
│   ├── agent_loop.py         # 主代理循环逻辑
│   ├── tools.py              # 工具函数实现
│   ├── background.py         # 后台任务管理
│   ├── notification.py       # 可恢复的统一通知队列
│   ├── session.py            # 主会话生命周期与运行编排
│   ├── session_state.py      # 会话与代理循环状态
│   ├── session_persistence.py # 会话检查点与可见消息事务
│   ├── subagent.py           # 子代理系统
│   ├── skill.py              # 技能注册和管理系统
│   ├── todo.py               # 待办事项管理
│   ├── compact.py            # 消息压缩优化
│   ├── cron.py               # 定时任务管理
│   ├── ai_config.py          # AI客户端配置
│   ├── directory.py          # 目录路径配置
│   ├── task.py               # 持久化任务管理
│   ├── permission/           # 权限管理模块
│   │   ├── __init__.py
│   │   ├── bash_security.py  # Bash安全验证器
│   │   └── permission.py     # 权限管理器
│   ├── memory/               # 记忆模块
│   │   ├── __init__.py
│   │   └── memory.py         # 记忆管理器
│   └── prompt/               # 提示词模块
│       ├── __init__.py
│       └── system-prompt.py  # 系统提示词管理器
├── .tasks/                   # 持久化任务存储目录（运行后生成）
└── skills/                   # 技能目录
```

## ⚙️ 配置要求

### 环境变量
需要设置API密钥环境变量（环境变量的名称应该和你的配置文件中指定的名称一致）：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 配置文件 (config.yaml)
```yaml
model:
  name: gpt-5.4
  base_url: https://api.openai.com/v1
  api_key_path_var: OPENAI_API_KEY

permission:
  mode: auto
  sandbox:
    # read=全部只读；workspace=仅当前工作目录可写；all=全部可写
    setting: workspace
    # false 时 Bubblewrap 会创建独立网络命名空间
    network-access: true
```

`auto` 模式仅支持 Linux，并要求系统已安装 `bwrap`（Bubblewrap）。沙箱的工作目录固定为程序启动时的当前目录。若 `bwrap` 不存在，命令会失败，不会自动降级为无沙箱执行。当命令因只读文件系统或权限限制失败时，程序会询问用户是否允许在非沙箱模式下重试；只有明确输入 `y` 或 `yes` 才会执行。

## 🎯 使用示例

### 基本交互
```bash
python src/main.py
>> 读取当前的项目，分析项目现有的功能，并在项目根目录下添加README.md介绍项目
```

### 会话持久化与恢复

会话、可见问答和最新 checkpoint 保存在 `.riton/db/agent.db`。无参数启动会创建并打印新的 `session_id`：

```bash
python src/main.py
```

继续旧会话或列出全部会话：

```bash
python src/main.py -s session_0123456789abcdef
python src/main.py -l
```

恢复会话时，程序会先恢复待处理通知、未完成的后台任务和工具调用，再逐行打印 `user`/`assistant` 历史并等待输入。后台任务和缺少结果的工具调用采用至少执行一次语义。Cron 与主 Agent 后台任务的完成通知会进入统一队列；即使 Agent 空闲，通知也会拉起新一轮处理，并把触发消息和最终回复写入可见历史。

### 权限管理使用
系统会自动在需要时请求权限确认：
```
  [Permission] bash with args {"command": "ls -la"}  
Allow? (y/n/always): y
```

### 待办事项管理
代理会自动创建和更新待办列表：
```
[~] 分析项目结构
[ ] 创建README.md
[o] 测试功能
2/3 completed
```

### 任务管理工具
系统现已支持持久化任务管理，可用于跨轮次保存任务状态与依赖关系：
- `task_create`：创建任务，可附带描述与 owner
- `task_get`：按 ID 查询任务详情
- `task_list`：列出当前所有任务及状态
- `task_update`：更新任务状态、owner，以及依赖/阻塞关系

任务数据保存在项目根目录下的`.tasks/`目录中，每个任务对应一个 JSON 文件。

### 运行调试
```bash
# 启用调试日志
python src/main.py --debug

# 正常运行
python src/main.py
```

Windows 上建议先创建虚拟环境并直接调用 Python；`start.sh` 面向 Linux：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe src\main.py -l
```

### 添加新技能
1. 在`skills/`目录下创建新技能文件夹
2. 创建`SKILL.md`文件，包含YAML前端元数据和Markdown主体
3. 系统会自动加载并注册新技能

### 工具开发
- 所有工具必须在`tools.py`中注册
- 工具函数需要处理异常并返回用户友好的错误信息
- 文件操作工具自动触发最近文件跟踪用于消息压缩

---

*RitonCode - 让AI成为你的智能编程伙伴*
