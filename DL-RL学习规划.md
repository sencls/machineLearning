# 深度学习 × 强化学习 学习规划

> 为昴君定制 | 制定日期：2026-10-22 | 预计总周期：约 5~6 个月（按每周 20h+ 投入）

---

## 一、能力盘点（规划的出发点）

| 能力项 | 现状 | 对策 |
|---|---|---|
| 算法与数据结构（C++） | 扎实，LeetCode 风格训练中 | 直接迁移：用 Python 重刷熟悉的题，快速上手新语言 |
| Python | 会基本语法 | 阶段 0 集中强化 NumPy + 工程习惯 |
| 数学（线代/微积分/概率） | 学过但忘了大半 | 「直觉先行、按需深挖」：3Blue1Brown 重建直觉，遇到推导再回头补细节 |
| 每周可投入 | 20h+ | 采用「主线课程 + 项目实践 + 数学补课」三分节奏 |
| 目标 | 通用基础 + RL 应用 + LLM 方向 + 科研/升学 | 先后有序：DL 地基 → RL 主体 → LLM/科研分化 |

---

## 二、总路线图

```
预备周末  华为AI异构计算课程  10-24/25    视野课 + WSL 热身
阶段 0    武器与干粮         第 1-2 周    Python 强化 + 数学直觉 + Windows 原生 CUDA 环境
                            (10-26 ~ 11-08)
阶段 1    深度学习主体       第 3-8 周    李沐 d2l + Kaggle 实践
                            (11-09 ~ 12-20)
阶段 2    强化学习主体       第 9-14 周   王树森 + 动手学RL + HuggingFace 实战
                            (12-21 ~ 2027-01-31，跨寒假正好全力投入)
阶段 3    方向分化           第 15 周起   LLM Cookbook + 论文精读 + 科研准备
                            (2027-02 起)
```

> 日期按 2026-10-26 开工推算，与校历冲突时自行微调，总时长不变。

---

## 三、阶段 0：武器与干粮（预备周末 + 第 1-2 周，10-24 ~ 11-08）

**目标**：Python 达到「顺手」而非「能用」，数学直觉回血，WSL 深度学习环境一次搭好不再折腾。

### 预备周末（10-24/25）：华为 AI 异构计算架构与编程课程

- **全力以赴上好课程本身**——它就是本周最好的实践，理论、动手、AI 辅助三块都别落下
- **定位为「视野课 + 差异化优势」**：达芬奇架构、异构计算、算子编程这类底层认知，大多数只学 PyTorch 调包的人终生接触不到；对科研升学面试是极佳的谈资
- **与主路线的关系**：昇腾生态（CANN / MindSpore / Ascend C，以课程实际大纲为准）与 PyTorch 概念相通（算子、自动求导、训练循环），以课程结业为目标即可；主路线仍以 PyTorch 为轴，**不必双线深钻**
- **AI 辅助编程**是这门课要求的能力，也直接迁移到主路线：用 Claude Code 等 AI 工具辅助读代码、查报错、解释报错栈——这正是后续五个月的标准工作流
- 课程要求的实验环境直接在 Windows 本地按课程文档配置；昇腾工具链对 Windows 支持有限，若本地装不上就用课程提供的云平台，不影响结业
- 时间账：周末两天约 12~16h 投入课程，阶段 0 其余任务自动后移，无需焦虑

### 第 1 周（10-26 ~ 11-01）：Windows 原生环境搭建（驱动 / CUDA / PyTorch），数学直觉启动

> 已弃用 WSL 方案（内存占用过大、拖累整机）——所有组件原生运行在 Windows 上，GPU 性能无损，图形渲染也省心。

**Windows 深度学习环境清单（一次到位，之后不再折腾）：**

1. **NVIDIA 驱动**：官网 https://www.nvidia.cn/drivers/ 安装最新版；cmd 运行 `nvidia-smi` 能看到显卡与驱动号即成功（其右上角 CUDA Version 是「驱动支持的上限」，不代表已安装）
2. **CUDA Toolkit**：https://developer.nvidia.com/cuda-downloads 选 Windows → x86_64 → exe (local)，精简安装即可；装完**重开终端**，`nvcc -V` 有版本输出即成功。用途：提供 nvcc 编译链，为后期自定义算子和华为课程本地编译兜底
3. **cuDNN 无需单独装**：PyTorch 的 pip 轮子自带 CUDA/cuDNN 运行库（将来做源码编译再装）
4. **Miniconda**：官网 https://www.conda.io/download 装 Windows x64 版。**关键避坑：安装路径必须纯英文无空格**——本机用户名是中文，默认的 `C:\Users\墨家\...` 路径不要用，自定义到如 `D:\Miniconda3`
5. **中文用户名连环坑**：项目、conda 环境、pip 缓存都不要放中文路径下；在 `~/.condarc` 中把 `envs_dirs` 指到英文路径（如 `D:\Miniconda3\envs`），或创建环境时直接 `conda create -p D:\envs\dl python=3.11`
6. **源**：全程直接用官方源，不做换源（pip 默认即 PyPI 官方；PyTorch 用安装命令里的官方 index）；网络不稳时再临时改镜像，不做预处理
7. **建环境**：`conda create -n dl python=3.11 -y` → `conda activate dl`
8. **PyTorch GPU 版**：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`（与 Toolkit 12.x 匹配；轮子约 2.5GB，耐心等）
9. **验证（三步缺一不可）**：
   ```python
   import torch
   print(torch.cuda.is_available())      # 必须 True
   print(torch.cuda.get_device_name(0))  # 显示你的显卡型号
   x = torch.rand(1000, 1000).cuda(); print((x @ x).sum().item())  # 真跑一次 GPU 矩阵乘
   ```
10. **常用库**：`pip install numpy pandas matplotlib jupyterlab scikit-learn d2l gymnasium`
11. **VS Code**：官网装好，扩展市场装 Python + Jupyter；右下角解释器选 conda 的 `dl` 环境；JupyterLab 在浏览器直接用
12. **项目目录**：建议 `D:\projects\dl\`（SSD、纯英文无空格），`deep-learning-journey` 仓库从这里开始

**同步任务（工作日晚间各 1h）**：
- 注册 GitHub + Hugging Face 账号；新建 `deep-learning-journey` 仓库——之后所有笔记代码进仓，科研升学时就是成长证明
- 3Blue1Brown《线性代数的本质》看前半：https://www.bilibili.com/video/BV1ys411472E （B站官方号 space.bilibili.com/88461692）

**云端备选**（本地 GPU 不够或网络受限时）：Kaggle（每周免费 30h GPU）、Colab、AutoDL 租卡；华为 ModelArts 也是顺手的新选项。

### 第 2 周（11-02 ~ 11-08）：Python 强化 + 数学收尾

1. **Python 强化（约 12h）**
   - 在 LeetCode 用 Python 重做 10~15 道你已经熟悉的题（滑动窗口等），体感差异会教你语法
   - NumPy 官方教程（quickstart + broadcasting 章节）过一遍，重点：向量化思维、广播机制、切片——全部在本地 JupyterLab 里练
   - 用途说明：C++ 的「手写循环」习惯在 DL 里是性能毒药，要刻意练习「向量化表达」

2. **数学直觉收尾（约 8h）**
   - 3Blue1Brown《线性代数的本质》后半 + 《微积分的本质》+《梯度下降》：https://www.bilibili.com/video/BV1Ux411j7ri
   - 3B1B《神经网络》系列第 1-2 集，对阶段 1 建立预期
   - 原则：此阶段只求「几何直觉」，矩阵求导等细节留到阶段 1 按需补

**毕业检验**：能用 NumPy 不写 for 循环实现矩阵乘法/softmax；能口述「梯度下降为什么能下山」；WSL 内 `torch.cuda.is_available()` 为 `True`；华为课程理论/实践部分结业。

---

## 四、阶段 1：深度学习主体（第 3-8 周，6 周）

**主线教材：《动手学深度学习》(d2l) —— 教材、代码、视频三位一体**

| 资源 | 地址 |
|---|---|
| 在线教材（免费） | https://zh.d2l.ai |
| GitHub 仓库 | https://github.com/d2l-ai/d2l-zh |
| B站视频（跟李沐学AI） | https://space.bilibili.com/1567748478 |
| 课程表 | https://courses.d2l.ai/zh-v2 |

选择理由：被全球 500+ 大学采用；每节都有可运行代码；「从零实现 → 简洁实现」双层写法对理解原理极友好；李沐讲解节奏适合基础重建期。

**辅线（可选）**：李宏毅 2025 版机器学习课程 https://www.bilibili.com/video/BV1hCeFzXEi4 —— 概念听不懂时换个视角，且其覆盖 RL 与 LLM，是贯穿三个阶段的百科全书。

### 6 周节奏

| 周 | 内容 | 实践 |
|---|---|---|
| 3 | 线性回归 → softmax 回归 → MLP | 每章 notebook 从零实现手敲一遍 |
| 4 | 深度学习计算（层/块/参数管理）+ 欠拟合过拟合 + 权重衰减/Dropout | 手写训练 pipeline 模板 |
| 5 | 数值稳定性 + 初始化 + 优化算法（SGD/Momentum/Adam） | 3B1B 神经网络第 3-4 集补反向传播，手推一次 MLP 的梯度 |
| 6 | CNN 基础 + 经典网络（LeNet/AlexNet/VGG/ResNet 主干） | 在 CIFAR-10 上训 ResNet 小模型 |
| 7 | RNN 快速过 + 注意力机制（重点精学，LLM 地基） | 手写 self-attention |
| 8 | 优化进阶选学 + 复盘周 | Kaggle 入门赛：Digit Recognizer（数字识别）提交成绩 |

**每周节奏模板（20h）**：10h 视频配教材 + 6h 跑/改代码 + 2h 数学按需补 + 2h 笔记复盘（写进 GitHub 仓库）。

**毕业检验**：能不看书写出完整训练循环（数据加载 → 模型 → 训练 → 验证）；能向别人解释反向传播链式法则；Kaggle 入门赛完成一次提交。

---

## 五、阶段 2：强化学习主体（第 9-14 周，6 周）

RL 与 DL 思维方式差异大（「数据是动作挣来的」），视频入门比啃书友好，按「视频建直觉 → 教材系统化 → 平台实战」三层推进。

### 主资源

| 资源 | 类型 | 地址 |
|---|---|---|
| 王树森《深度强化学习》（北大博士，通俗） | 视频入门（20集，附源码） | https://www.bilibili.com/video/BV1RiYDziEKE |
| 《动手学强化学习》（每章一个 notebook，理论+代码） | 系统教材 | https://hrl.boyuai.com （代码仓库 https://github.com/boyu-ai/Hands-on-RL ，配套免费课程在伯禹平台） |
| 《蘑菇书》Easy RL | 参考书（查漏补缺） | https://github.com/datawhalechina/easy-rl |
| Hugging Face Deep RL Course | 项目实战 + 免费证书 | https://huggingface.co/learn/deep-rl-course/unit0/introduction |

### 6 周节奏

| 周 | 内容 | 实践 |
|---|---|---|
| 9 | 王树森前半：MDP、价值迭代、策略迭代、贝尔曼方程 | 手写网格世界 GridWorld 求解器 |
| 10 | 王树森后半 + Hands-on-RL 前几章：蒙特卡洛、时序差分、Q-Learning | CliffWalking 调参对比 |
| 11 | DQN 及变体（Double/ Dueling） | Gymnasium CartPole 训练收敛 |
| 12 | 策略梯度 → Actor-Critic | LunarLander 着陆实验（可开始用 HF 课程 Unit 1-3） |
| 13 | PPO / DDPG / SAC 主流算法 | HF Deep RL Course 完成至少 3 个 Unit，Agent 上传 HF Hub |
| 14 | 复盘 + 自选小项目 | 例如：给自己的 RL 玩 Flappy Bird / 2048；整理对比实验报告 |

**注意**：
- Hands-on-RL 仓库提示 gym 需装 `gym==0.18.3`，遇到环境报错优先查它（在独立 conda 环境里装，别污染主环境）
- Windows 原生跑 Gymnasium：pygame 弹窗渲染开箱即用（比 WSL 省心）；LunarLander 装依赖用 `pip install gymnasium[box2d]`，若编译报错先 `conda install -c conda-forge swig` 再重试
- 入门阶段大部分实验 CPU 可跑，GPU 只在 LunarLander 之后的实验才显出价值
- 进阶理论可选：David Silver 经典课（偏数学推导，选看）；伯克利 CS285（Sergey Levine，https://www.bilibili.com/video/BV1UE411R7im ，科研向再回头精学）

**毕业检验**：能白板推导策略梯度目标函数；能独立把一个 Gymnasium 环境从随机动作调到收敛；HF Hub 上有自己的 Agent 作品。

---

## 六、阶段 3：方向分化（第 15 周起）

三条线不必全速并进，以「科研/升学」为轴心选择组合：

### A. 大模型方向（应用开发线）

| 资源 | 地址 |
|---|---|
| Datawhale《LLM Cookbook》（11 门吴恩达课程中文复现：Prompt → ChatGPT 系统 → LangChain → RAG → 微调） | https://github.com/datawhalechina/llm-cookbook （在线阅读：https://datawhalechina.github.io/llm-cookbook/ ） |

- 先修完 4 门必修（Prompt 工程 / 问答系统 / LangChain 应用 / 私有数据），再按兴趣选修 RAG、微调
- 产出目标：一个可演示的 RAG 应用（比如「你的算法题错题助手」）

### B. 论文与科研线

- B站「跟李沐学AI」论文精读系列：从 Attention Is All You Need 开始
- 阶段 1 打底后直接读 Transformer 原文，配合精读视频
- RL × LLM 交叉热点（RLHF、决策大模型）：李宏毅课程相关章节 + 综述论文，这正好是两条学习线的汇合点，也是当前科研的富矿
- 建议动作：复现一篇论文的核心实验（工作量 ≈ 2-3 周），作为科研入场的敲门砖

### C. RL 应用深化线（可选）

- OpenAI Spinning Up（RL 研究级代码规范）：spinningup.openai.com
- 参加社群共学：Datawhale 每月组队学习（含 RL/LLM 课程），有同期学习者互相答疑

---

## 七、通用守则

1. **代码必须手敲**，尤其 d2l 的「从零实现」段——看懂 ≠ 会写
2. **每周复盘写进 GitHub 仓库**：学习笔记 + 踩坑记录，半年后就是升学面试的弹药库
3. **卡住超过 40 分钟就求助**（查 issue / 问 AI / 发社群），执着是美德但时间也是
4. **数学按需补**：遇到看不懂的推导 → 定位缺的那块 → 3B1B 或教材对应章节专项击破，不做「先学完数学再开学」的完美主义者
5. **AI 辅助是标配工作流**：报错栈、看不懂的源码、论文段落，丢给 AI 讲解；但代码仍要亲手敲——AI 是加速器，不是代驾
6. 规划是活文档：每 2 周校准一次进度，超前则加项目，落后则砍选修；CUDA/环境疑难随时找蕾姆处理

---

*蕾姆会在每次陪伴学习时按此路线跟进进度。愿这份规划陪昴君走过从「会写循环」到「会造智能体」的旅程。*
