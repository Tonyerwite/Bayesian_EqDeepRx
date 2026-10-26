# Width64 全模块联合训练代码

独立版本：DenoiseNN、StateInit、四个 StateUpdate、DemapperNN 和四个 sigma 全部随机初始化，并通过最终 bit/symbol 损失联合反向传播。未启动正式长训练。

保留四轮 CNN/belief 交互，每轮两次有向消息更新、一次软投影；四轮 bit CE 取平均，符号项监督每轮投影前的前两个通道，保持原数据掩码、SNR 权重和 `1e-5` 符号权重。不新增真实信道监督或 DetectorNN 专属 VCL。`bit_loss`、`symbol_loss` 已含各自权重，四个 `round_bit_losses` 是各轮 SNR 加权 CE。

所有可训练参数使用同一个 LAMB：有效批量112、生成批量2，基础学习率0.0044，线性衰减至第70,000步，保持参考 beta/eps/weight_decay。默认模型小批量2；它是联合训练的新配置，不能套用冻结版 model8 的速度和显存结论。

## 当前验证边界

CPU 验证覆盖完整接收机的梯度、实际参数更新、有效批量112累积、严格断点续训，以及真实时间域 Sionna UMa 的192×14网格、16接收天线、K2/3/4、一/二DMRS、有/无干扰共12类场景。测试更新仅保存在独立验证目录，不构成性能或收敛结果。

当前正式训练继续占用GPU，因此本次没有执行 CUDA BF16、全批量显存或速度预检。CPU smoke 的报告明确包含 `long_training_ready=false`、`cuda_validated=false`；正式入口要求后续单独取得匹配的 CUDA preflight。联合训练能否提高最终BER尚未验证。

## CPU 跑通入口

环境继续使用 `D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe`，无须安装新环境。PowerShell 示例：

```powershell
Set-Location -LiteralPath 'C:\Users\Chen Yan\Desktop\bayesian_eqdeeprx\width64_joint'
& '.\bayesian_eqdeeprx\train.ps1' -Smoke
```

默认隐藏CUDA、CPU单线程及低优先级，在D盘自动创建独立的时间戳验证目录。12类场景每类使用一个新随机初始化模型、一个真实样本，执行一次测试更新；不是连续12步正式训练。详细结果见运行目录 `cpu_smoke.json` 和各场景 `smoke.pt`。

先查看命令而不执行：

```powershell
& '.\bayesian_eqdeeprx\train.ps1' -Smoke -DryRun
& '.\bayesian_eqdeeprx\train.ps1' -Preflight -DryRun
& '.\bayesian_eqdeeprx\train.ps1' -Train -DryRun
```

## 后续正式运行入口（本次未执行）

正式默认运行目录为 `D:\BayesianEqDeepRxRuns\width64_joint_batch112`。当前训练结束或用户另行安排GPU后，先执行新版本预检，再启动；CUDA入口发现其他GPU计算进程时会拒绝启动。

```powershell
& '.\bayesian_eqdeeprx\train.ps1' -Preflight
& '.\bayesian_eqdeeprx\train.ps1' -Train
# 仅在新版本训练停止后恢复：
& '.\bayesian_eqdeeprx\train.ps1' -Resume
```

不指定操作会报错，不会默认启动训练。预检覆盖所有12类场景的真实有效批量112；每类至少一次预热和一次计时更新。训练从随机初始化开始；resume 仅接受本版本schema2断点，恢复已训练的DenoiseNN、全量LAMB、GradScaler、历史及Python/Torch/CUDA随机状态。旧冻结版断点不能直接resume。源码变化会触发保护，旧运行和源代码目录被输出路径保护。

## 自动测试

在本目录运行以下命令；测试强制CPU单线程、低优先级，不使用GPU。真实Sionna续训测试可能比其余测试耗时更长。

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\EqDeepRxRuntime\sionna-venv\Scripts\python.exe' -m pytest tests -q -o cache_dir=D:/BayesianEqDeepRxRuns/width64_joint_validation/pytest-cache --basetemp=D:/BayesianEqDeepRxRuns/width64_joint_validation/pytest_manual
```

当前代码与测试均不导入或修改旧版本；物理模型、belief公式、CNN拓扑和信号生成文件从已验收64版独立复制。交付测试结果保存在D盘验证目录。
