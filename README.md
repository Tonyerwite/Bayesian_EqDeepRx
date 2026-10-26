# Bayesian EqDeepRx — Width 64

本仓库包含两套完整、独立的宽度64接收机及训练代码。

| 版本 | 目录 | DenoiseNN | 验证与训练状态 |
|---|---|---|---|
| 当前正式训练版本 | [width64_fast](width64_fast/README.md) | 加载随代码提供的预训练资产并冻结 | 已验证 CUDA 训练配置，正在进行70,000步训练 |
| 全模块联合训练版本 | [width64_joint](width64_joint/README.md) | 随机初始化，随最终 bit/symbol 损失共同训练 | CPU 单元测试、真实 Sionna 场景及严格续训已验证；尚未进行 CUDA 预检和正式长训练 |

两个目录均包含完整的前端信道处理、LMMSE/RZF、StateInit、四个 StateUpdate、belief 消息更新与软投影、DemapperNN、损失函数、Sionna 信号生成、LAMB 训练、断点续训和 PowerShell 入口。

## 每步训练量

两版默认有效批量均为 **112个时隙样本**，每步执行一次 LAMB 更新。每个时隙采用192×14资源格、16根接收天线，随机选择2、3或4个用户、一/二 DMRS、有/无额外干扰。

| 配置 | 冻结 DenoiseNN 版本 | 联合训练版本 |
|---|---:|---:|
| 有效批量 | 112 | 112 |
| 每次生成样本数 | 2 | 2 |
| 每步数据生成次数 | 56 | 56 |
| 模型小批量 | 8 | 2 |
| 每步梯度累积次数 | 14 | 56 |
| RE 消息分块 | 8192 | 8192 |
| 目标优化器步数 | 70000 | 70000 |

模型内部的四轮 CNN/belief 交互包含在上述前向与反向中，不增加独立样本数。联合训练版本的速度和显存尚未验证，不能沿用冻结版的测速结论。

## 环境与入口

已验证环境为 Python 3.12、PyTorch 2.11 CUDA 12.8、Sionna 2.1；依赖声明分别位于两个版本的 `bayesian_eqdeeprx/pyproject.toml`。两版使用相同的 Python 导入包名，请从各自版本目录运行，或使用独立虚拟环境。

脚本默认 Python 路径为 `D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe`，运行文件默认写入 `D:\BayesianEqDeepRxRuns`。其他机器通过 `-PythonPath` 和 `-RunDir` 指定实际位置。

### 冻结 DenoiseNN 版本

```powershell
Set-Location .\width64_fast
# 新训练：先在独立运行目录完成 CUDA 预检，再启动训练。
& .\bayesian_eqdeeprx\train.ps1 -Preflight -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
& .\bayesian_eqdeeprx\train.ps1 -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
# 已有完整断点的续训：
& .\bayesian_eqdeeprx\train.ps1 -Resume -RunDir 'D:\BayesianEqDeepRxRuns\my_width64_run'
```

`pretrained_denoiser.pt` 是本版本必需的固定前端资产，已纳入仓库。训练生成的模型断点、优化器状态、日志和缓存保存在运行目录，不随源码发布。启动前应确认没有同一任务的其他训练进程，避免重复训练。

### 联合训练 DenoiseNN 版本

```powershell
Set-Location .\width64_joint
# CPU 跑通；使用独立输出目录。
& .\bayesian_eqdeeprx\train.ps1 -Smoke
# 查看命令，不执行：
& .\bayesian_eqdeeprx\train.ps1 -Train -DryRun
# GPU 空闲时，先预检，再正式训练：
& .\bayesian_eqdeeprx\train.ps1 -Preflight
& .\bayesian_eqdeeprx\train.ps1 -Train
# 本版本完整断点续训：
& .\bayesian_eqdeeprx\train.ps1 -Resume
```

联合版通过四轮 bit CE 和投影前软符号损失联合训练 DenoiseNN、StateInit、四个 StateUpdate、DemapperNN 和 sigma。冻结版与联合版的断点协议不同，不能直接互相续训。测试命令及验证边界见 [联合版说明](width64_joint/README.md)。

## 文件一致性与验证记录

- 冻结版源码指纹：`6e540e10b99014ad718ba57a488e684ee122320a72d55d09641803cbc0310b89`。
- 联合版源码指纹：`df31da2cb5bfbb7161915c4fc25724b858fb137e5797da089e83a888f4becfe3`。
- 各版本的 `CODE_MANIFEST.json` 保存逐文件校验信息；冻结版另有 `RUN_PROFILE.json` 和 `ACCEPTANCE.md`。
- 上述交付记录中的282步、训练未启动等字段是2026年10月24日验收时的历史快照。实际训练状态以外部运行目录中的日志、完整断点和启动记录为准，GitHub 不提供实时进度。
- `.gitattributes` 保留原文件字节与换行，避免 Git 换行转换使源码指纹失效。

仓库仅跟踪这两套64宽度代码及相关说明；论文、旧大模型、历史实验、运行断点、日志及缓存保留在本地。
