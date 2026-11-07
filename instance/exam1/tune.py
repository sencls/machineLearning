# -*- coding: utf-8 -*-
"""超参数调优脚本(网络固定: Linear(in,64) -> ReLU -> Linear(64,1))

方法:
  1. 5折交叉验证评估每个超参数组合(与Kaggle真实分数已验证高度一致)
  2. 逐epoch监控验证误差,连续patience轮无提升则早停
  3. 用 train-valid gap 诊断过拟合/欠拟合:
       gap大 + train低  -> 过拟合(靠早停与weight_decay压制)
       train/valid双高  -> 欠拟合(需更大lr或容量,本脚本网络不变只调lr)
输出: tuning_results.csv / best_fold0_curves.png / submission.csv
"""
import torch                      # Windows铁律: torch 必须第一个导入
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
from d2l import torch as d2l

ROOT = Path(__file__).resolve().parent.parent        # D:\machineLearning
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent                # D:\machineLearning\exam1

# ---------- 1. 数据预处理(与 instance/exam1.py 完全一致) ----------
train_data = pd.read_csv(DATA / "kaggle_house_pred_train.csv")
test_data = pd.read_csv(DATA / "kaggle_house_pred_test.csv")
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

num = all_features.dtypes[all_features.dtypes != "object"].index
all_features[num] = all_features[num].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[num] = all_features[num].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=float)

n_train = train_data.shape[0]
X = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
y = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
in_features = X.shape[1]

loss = nn.MSELoss()


def log_rmse(net, f, l):
    """Kaggle官方指标: 对数价格空间的RMSE(预测值clamp到>=1保证log有效)"""
    c = torch.clamp(net(f), 1, float("inf"))
    return torch.sqrt(loss(torch.log(c), torch.log(l))).item()


# ---------- 2. 固定网络(本脚本只调超参数) ----------
def get_net():
    return nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))


# ---------- 3. 带早停的训练 ----------
def train_es(net, Xt, yt, Xv, yv, lr, wd, bs, max_epochs, patience):
    """训练并逐epoch计算验证误差;连续patience轮无提升则提前终止。
    返回最优epoch处的(train, valid)指标、最优轮数、实际运行轮数、曲线"""
    it = d2l.load_array((Xt, yt), bs)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best_v, best_t, best_ep = float("inf"), float("inf"), 0
    no_improve, ran = 0, 0
    tls, vls = [], []
    for ep in range(1, max_epochs + 1):
        for xb, yb in it:
            opt.zero_grad()
            loss(net(xb), yb).backward()
            opt.step()
        ran = ep
        v = log_rmse(net, Xv, yv)
        vls.append(v)
        tls.append(log_rmse(net, Xt, yt))
        if v < best_v - 1e-4:            # 有实质提升才重置计数
            best_v, best_t, best_ep = v, tls[-1], ep
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_t, best_v, best_ep, ran, tls, vls


# ---------- 4. K折评估 ----------
def k_fold_es(lr, wd, bs, k=5, max_epochs=300, patience=30):
    fs = X.shape[0] // k
    ts, vs, eps = [], [], []
    fold0_curves = None
    for i in range(k):
        torch.manual_seed(7)            # 每折固定种子: 各配置公平对比
        idx = slice(i * fs, (i + 1) * fs)
        mask = torch.ones(X.shape[0], dtype=torch.bool)
        mask[idx] = False
        t, v, ep, ran, tls, vls = train_es(
            get_net(), X[mask], y[mask], X[idx], y[idx],
            lr, wd, bs, max_epochs, patience)
        ts.append(t)
        vs.append(v)
        eps.append(ep)
        if i == 0:
            fold0_curves = (tls, vls, ep, ran)
    return float(np.mean(ts)), float(np.mean(vs)), float(np.mean(eps)), fold0_curves


# ---------- 5. 网格搜索 ----------
rows = []
grid_lr = [0.5, 1, 2, 4]
grid_wd = [0.0, 0.01, 0.05, 0.1]
for lr in grid_lr:
    for wd in grid_wd:
        t, v, ep, _ = k_fold_es(lr, wd, bs=64)
        gap = v - t
        diag = "OVERFIT" if gap > 0.05 else ("UNDERFIT" if v > 0.20 else "ok")
        rows.append(dict(lr=lr, wd=wd, train=t, valid=v, gap=gap,
                         best_epoch=int(ep), diag=diag))
        print(f"lr={lr:<4} wd={wd:<5} train={t:.4f} valid={v:.4f} "
              f"gap={gap:+.4f} stop_ep={int(ep):>3} {diag}")

df = pd.DataFrame(rows).sort_values("valid").reset_index(drop=True)
df.to_csv(OUT / "tuning_results.csv", index=False)
best = df.iloc[0]
print(f"\nBEST: lr={best.lr} wd={best.wd} valid={best.valid:.4f} "
      f"train={best.train:.4f} gap={best.gap:+.4f} best_epoch={best.best_epoch}")

# ---------- 6. 最优配置 fold-0 曲线(可视化早停位置) ----------
_, _, _, curves = k_fold_es(float(best.lr), float(best.wd), bs=64)
tls, vls, best_ep, ran = curves
d2l.plt.figure()
d2l.plt.plot(range(1, len(tls) + 1), tls, label="train")
d2l.plt.plot(range(1, len(vls) + 1), vls, label="valid")
d2l.plt.axvline(best_ep, color="gray", linestyle="--",
                label=f"best epoch={best_ep}")
d2l.plt.xlabel("epoch")
d2l.plt.ylabel("log rmse")
d2l.plt.legend()
d2l.plt.title(f"lr={best.lr} wd={best.wd} (stopped at {ran})")
d2l.plt.savefig(OUT / "best_fold0_curves.png", dpi=150, bbox_inches="tight")
d2l.plt.close()

# ---------- 7. 用最优配置全量训练(固定为K折最优轮数)并生成提交 ----------
n_final = int(best.best_epoch)
torch.manual_seed(7)
net = get_net()
it = d2l.load_array((X, y), 64)
opt = torch.optim.Adam(net.parameters(), lr=float(best.lr),
                       weight_decay=float(best.wd))
for _ in range(n_final):
    for xb, yb in it:
        opt.zero_grad()
        loss(net(xb), yb).backward()
        opt.step()

preds = net(X_test).detach().numpy().reshape(-1)
test_data["SalePrice"] = pd.Series(preds)
pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1).to_csv(
    OUT / "submission.csv", index=False)
print(f"final model: {n_final} epochs on full data -> submission.csv saved")
