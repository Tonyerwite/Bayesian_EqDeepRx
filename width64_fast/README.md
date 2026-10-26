# 最快width64完整训练版本

**算法、公式、精度和训练原理保持原样。正式训练已于2026-10-24 22:19（北京时间）从282步完整断点恢复，每2小时 steps/loss 监督已启用。**

本版有效批量112、每次生成2、模型拼批8、RE分块8192；模型投影保持原eager运算。信道生成只调整索引/搬运顺序，并捕获原ATen几何运算为CUDA Graph，没有替换计算公式。

实际连续混合窗口为 **10.67秒/步，算术外推约675步/两小时**；按原12种场景概率重加权为 **10.30秒/步，约699步/两小时**。完整结果和边界见 [ACCEPTANCE.md](ACCEPTANCE.md)。

## 代码与训练文件

- `bayesian_eqdeeprx/`：完整、独立的64版运行代码和固定DenoiseNN资产。
- `RUN_PROFILE.json`：已验收的运行配置，不是训练超参数调优结果。
- `CODE_MANIFEST.json`：交付文件指纹和续训准备说明。
- 正式训练目录：`D:\BayesianEqDeepRxRuns\width64_fast_batch112`。
- 现有准备断点：282步，来自原64版正式断点；未采用速度实验中更新到366步的权重。

公共运行环境继续使用 `D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe`，不需要安装或更换环境。训练缓存、日志和断点继续留在D盘。

## 正式训练与续训

从本目录调用已配置好的入口：

```powershell
Set-Location -LiteralPath 'C:\Users\Chen Yan\Desktop\bayesian_eqdeeprx\width64_fast'
& '.\bayesian_eqdeeprx\train.ps1' -Resume
```

本次已执行上述命令，从282步恢复，目标总步数70,000，有效批量112、生成批量2；学习率进度、LAMB、GradScaler及随机状态均从完整断点恢复。再次续训时将从运行目录最新完整断点恢复；执行前应确认当前训练已经停止，避免启动重复进程。实时进度见D盘运行目录的 `training.jsonl`，启动记录见 `launch.json`。

如需重新做运行环境预检，可在得到相应指令后运行：

```powershell
& '.\bayesian_eqdeeprx\train.ps1' -Preflight -Repeats 3
```

首次运行会重建CUDA Graph并预热，不能要求冷启动第一步等于稳态平均速度。64版监督已更新到本代码目录和新D盘运行目录，每2小时报告 steps/loss、速度与异常。旧大模型监督保持暂停。`ACCEPTANCE.md`、`CODE_MANIFEST.json`及`RUN_PROFILE.json`中的启动前记录为交付验收时的历史状态。
