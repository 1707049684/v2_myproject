# GNCDM Project Structure

```
GNCDM/
|
|-- core/                           # 核心模块（G-NCDM 原有代码）
|   |-- __init__.py
|   |-- model.py                    # G-NCDM 模型 + expand_topology()
|   |-- train.py                    # 训练/评估 + IDCDataset + Hybrid DataLoader
|   |-- loss.py                     # BCE 损失函数
|   |-- tools.py                    # 工具函数
|   |-- diagnose.py                  # 诊断功能
|   |-- reliability.py               # 可靠性分析
|   |-- model_parser.py             # 模型解析器
|   |-- run.py                      # 运行入口
|   |-- run_ae.py                   # AE 运行入口
|
|-- incremental/                    # 增量学习模块（本次新增）
|   |-- __init__.py
|   |-- loss.py                     # TopologyAwareDecoupledLoss 解耦损失
|   |-- metrics.py                  # TMD 计算器 + LR 预热调度器
|
|-- experiments/                    # 实验脚本
|   |-- test_incremental.py         # 增量学习单元测试
|   |-- run_ablation_comparison.py  # 消融对比实验
|
|-- data/                          # 数据文件
|   |-- *.csv                       # 训练/测试数据
|   |-- *_Q_matrix.npy              # Q 矩阵
|
|-- config/                        # 配置文件
|   |-- training_config_*.json      # 训练配置
|
|-- scripts/                       # Shell 脚本
|   |-- gncdm_*.sh                  # 训练/诊断脚本
|
|-- result/                        # 输出结果
|
|-- __init__.py                    # 包初始化
|-- requirements.txt               # 依赖包
|-- README.md                     # 项目说明
|-- PROJECT_STRUCTURE.md          # 项目结构说明
|-- ablation_results.csv          # 消融实验结果
```

## 模块说明

### 1. core/ - 核心模块
原有 G-NCDM 模型的核心实现：
- **model.py**: 模型定义 + 增量扩展方法 `expand_topology()`
- **train.py**: 数据加载器、训练/评估函数
- **loss.py**: BCE 损失函数

### 2. incremental/ - 增量学习模块
本次改造新增的增量学习组件：
- **loss.py**: 解耦蒸馏损失 `TopologyAwareDecoupledLoss`
- **metrics.py**: 评估指标和调度器（TMD、LR Warm-up）

### 3. experiments/ - 实验脚本
用于验证和测试的脚本：
- **test_incremental.py**: 单元测试
- **run_ablation_comparison.py**: 消融对比实验

## 使用方法

### 增量学习训练示例
```python
from core.model import GNCDM
from incremental.loss import TopologyAwareDecoupledLoss
from incremental.metrics import calculate_rd, LinearWarmupScheduler
from core.train import AsymmetricHybridDataLoader

# 1. 加载旧模型
model_old = GNCDM(...)

# 2. 扩展拓扑
model_dynamic.expand_topology(delta_M, delta_K, Q_expanded)

# 3. 配置增量训练
loss_fn = TopologyAwareDecoupledLoss(model_old, model_dynamic, n_know_old)
hybrid_loader = AsymmetricHybridDataLoader(old_loader, new_loader)

# 4. 训练和评估
train_incremental(model_dynamic, hybrid_loader, loss_fn, ...)
```

### 运行消融实验
```bash
cd GNCDM
python experiments/run_ablation_comparison.py
```

### 运行单元测试
```bash
cd GNCDM
python experiments/test_incremental.py
```

## 导入路径

| 模块 | 导入语句 |
|------|---------|
| G-NCDM 模型 | `from core.model import GNCDM` |
| 训练函数 | `from core.train import train, eval_incremental` |
| 解耦损失 | `from incremental.loss import TopologyAwareDecoupledLoss` |
| RD 计算器 | `from incremental.metrics import calculate_rd` |
| LR 预热 | `from incremental.metrics import LinearWarmupScheduler` |
