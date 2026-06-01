import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Avalanche 组件
from avalanche.benchmarks import dataset_benchmark
from avalanche.training.supervised import DER

# 引入认知诊断核心评测算子
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score

# ==========================================
# 1. 认知诊断专用的基础骨干网络 (Backbone)
# ==========================================
class CognitiveBackbone(nn.Module):
    def __init__(self, num_students, num_items, embed_dim=64):
        super(CognitiveBackbone, self).__init__()
        self.student_emb = nn.Embedding(num_students, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        stu_idx = x[:, 0]
        item_idx = x[:, 1]
        u = self.student_emb(stu_idx)
        v = self.item_emb(item_idx)
        cat_features = torch.cat([u, v], dim=-1)
        logits = self.mlp(cat_features)
        return logits

# ==========================================
# 2. 测量学专用评测函数 (拦截计算 AUC/RMSE/ACC/F1)
# ==========================================
def evaluate_cd_metrics(model, test_dataset, device):
    """提取单个任务的 CD 指标"""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    y_trues, y_probs, y_preds = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1] # 取预测为 1 的概率
            preds = torch.argmax(logits, dim=1)
            
            y_trues.extend(y.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            y_preds.extend(preds.cpu().numpy())
            
    # 防止单个 batch 中只有一类标签导致 AUC 报错
    try:
        auc = roc_auc_score(y_trues, y_probs)
    except ValueError:
        auc = 0.5 
        
    rmse = mean_squared_error(y_trues, y_probs)**0.5
    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, zero_division=0)
    
    return auc, rmse, acc, f1

# ==========================================
# 3. 数据流装载：严格执行 2/3(旧) 与 1/3(新) 的物理隔离划分
# ==========================================
def load_math1_strict_partition(csv_path=r"C:\Users\Administrator\WPSDrive\1657690587\WPS企业云盘\深圳大学\我的企业文档\Generative-CD-main\GNCDM\data\math1_train_0.8_0.2.csv"):
    print(f">>> 正在从 {csv_path} 加载真实科研数据...")
    df = pd.read_csv(csv_path)
    
    user_col = 'user_id' if 'user_id' in df.columns else 'student_id'
    item_col = 'item_id' if 'item_id' in df.columns else 'question_id'
    score_col = 'score' if 'score' in df.columns else 'correct'
    
    # 1. 提取全局唯一题目集合，并进行随机打乱（消除人为排序偏误）
    unique_items = df[item_col].unique()
    np.random.seed(42) # 固定随机种子保证每次跑出来的切分一致
    np.random.shuffle(unique_items)
    
    # 2. 严格按 2/3 和 1/3 划分题目集合
    split_idx = int(len(unique_items) * (2/3))
    old_items_set = set(unique_items[:split_idx])
    new_items_set = set(unique_items[split_idx:])
    print(f">>> 题目拓扑切分完成: 旧题目(Task 0) = {len(old_items_set)} 个, 新题目(Task 1) = {len(new_items_set)} 个")

    # 3. 剥离数据框
    df_old = df[df[item_col].isin(old_items_set)]
    df_new = df[df[item_col].isin(new_items_set)]
    
    train_datasets, test_datasets = [], []
    
    # 4. 构建 Avalanche 的双阶段任务流
    for task_df in [df_old, df_new]:
        x_task = torch.tensor(task_df[[user_col, item_col]].values, dtype=torch.long)
        y_task = torch.tensor(task_df[score_col].values, dtype=torch.long)
        
        # 任务内的 8:2 随机划分 (Train / Test)
        indices = np.random.permutation(len(x_task))
        split_point = int(len(x_task) * 0.8)
        train_idx, test_idx = indices[:split_point], indices[split_point:]
        
        train_datasets.append(TensorDataset(x_task[train_idx], y_task[train_idx]))
        test_datasets.append(TensorDataset(x_task[test_idx], y_task[test_idx]))

    return dataset_benchmark(train_datasets=train_datasets, test_datasets=test_datasets)

# ==========================================
# 4. 核心训练流与决战数据提取
# ==========================================
def run_der_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tasks = 2 # 严格锁定为 2 阶段：[0: 2/3旧题基线], [1: 1/3新题增量]
    
    benchmark = load_math1_strict_partition()
    
    model = CognitiveBackbone(num_students=5000, num_items=5000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    cl_strategy = DER(
        model, optimizer, criterion,
        mem_size=500, alpha=0.1, beta=0.5, 
        train_mb_size=128, train_epochs=10, 
        eval_mb_size=256, device=device
    )

    print("\n>>> 启动 DER++ 增量训练...")
    
    # 追踪矩阵：用于计算你定义的流形欧式空间 TMD
    baseline_student_emb = None 
    peak_auc_history = {}

    for experience in benchmark.train_stream:
        task_id = experience.current_experience
        task_name = "【旧知识基线】" if task_id == 0 else "【新知识增量】"
        print(f"\n--- 开始训练 Task {task_id} {task_name} ---")
        cl_strategy.train(experience)
        
        # 记录当前任务刚学完时的巅峰 AUC
        curr_test_ds = benchmark.test_stream[task_id].dataset
        auc, _, _, _ = evaluate_cd_metrics(model, curr_test_ds, device)
        peak_auc_history[task_id] = auc
        print(f"Task {task_id} 收敛后巅峰 AUC: {auc:.4f}")
        
        # 获取 Baseline 锚点 (T0 状态的特质向量)
        if task_id == 0:
            # 必须用 clone().detach() 将其从计算图中硬隔离出来并缓存
            baseline_student_emb = model.student_emb.weight.data.clone().cpu()

    print("\n" + "="*60)
    print(" 终极评估：生成论文决战表格数据")
    print("="*60)
    
    # 1. 计算 _new 指标 (即在刚学完的 Task 1 上的性能)
    new_dataset = benchmark.test_stream[1].dataset
    auc_new, rmse_new, acc_new, f1_new = evaluate_cd_metrics(model, new_dataset, device)
    
    # 2. 计算 _old 指标 (即学习完 Task 1 后，回过头去考 Task 0 旧题的性能)
    old_ds = benchmark.test_stream[0].dataset
    auc_old, rmse_old, acc_old, f1_old = evaluate_cd_metrics(model, old_ds, device)
    
    # =========================================================================
    # 3. 核心计算：基于几何流形欧氏距离的 TMD (Theoretical Maximum Disruption)
    # =========================================================================
    # 获取增量学习收敛后的特质向量 (T1 状态的 \hat{\theta}_{i})
    final_student_emb = model.student_emb.weight.data.cpu()
    
    # 严格对齐 Ours (DynamicDNA) 的 TMD 计算公式：
    # TMD = mean(||theta_old - theta_new[:K_old]||_2 / sqrt(K_old))
    K_old = 7  # math1 旧知识维度
    final_student_emb_old_dim = final_student_emb[:, :K_old]
    baseline_student_emb_old_dim = baseline_student_emb[:, :K_old]
    tmd_euclidean_distances = torch.norm(baseline_student_emb_old_dim - final_student_emb_old_dim, p=2, dim=1)
    tmd = (tmd_euclidean_distances / np.sqrt(K_old)).mean().item()
    # =========================================================================

    # 打印最终可直接填入论文的 Markdown 表格
    print("\n请直接将以下数据复制填入你的论文表格中：\n")
    print("| Model | AUC_old | AUC_new | RMSE_old | RMSE_new | ACC_old | ACC_new | F1_old | F1_new | TMD (Euclidean Distance) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(f"| DER++ | {auc_old:.4f} | {auc_new:.4f} | {rmse_old:.4f} | {rmse_new:.4f} | {acc_old:.4f} | {acc_new:.4f} | {f1_old:.4f} | {f1_new:.4f} | {tmd:.4f} |")
    print("\n" + "="*60)

if __name__ == '__main__':
    run_der_baseline()