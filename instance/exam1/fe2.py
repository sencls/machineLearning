# -*- coding: utf-8 -*-
"""特征工程第二轮: 修正评估协议

fe.py 的教训: 标签取对数后,参数解的量级从~18万缩到~12,
原协议(lr网格偏高、patience=30太短)对L1~L3不公平:
  - lr>=0.5 在对数标签下直接发散(fe.py实测 train=259)
  - 小lr下输出偏置要走~12/lr步,patience=30不够它收敛
本轮: lr网格下沉到0.01~0.2, max_epochs=600, patience=60
基线参照: fe.py L0 = 0.1317(lr=4, 原始标签)
"""
import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
from d2l import torch as d2l

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent

train_data = pd.read_csv(DATA / "kaggle_house_pred_train.csv")
test_data = pd.read_csv(DATA / "kaggle_house_pred_test.csv")
n_train = train_data.shape[0]
labels_raw = train_data.SalePrice.values.reshape(-1, 1).astype("float32")


def build_features(level):                          # 与 fe.py 完全一致
    af = pd.concat((train_data.iloc[:, 1:-1].copy(), test_data.iloc[:, 1:].copy()))
    if level >= 2:
        af["TotalSF"] = (af["TotalBsmtSF"].fillna(0) + af["1stFlrSF"].fillna(0)
                         + af["2ndFlrSF"].fillna(0))
        af["TotalBath"] = (af["FullBath"].fillna(0) + 0.5 * af["HalfBath"].fillna(0)
                           + af["BsmtFullBath"].fillna(0)
                           + 0.5 * af["BsmtHalfBath"].fillna(0))
        af["Age"] = af["YrSold"] - af["YearBuilt"]
        af["RemodAge"] = af["YrSold"] - af["YearRemodAdd"]
        af["QualSF"] = af["OverallQual"] * af["GrLivArea"]
        af["MSSubClass"] = af["MSSubClass"].astype(str)
    if level >= 3:
        qmap = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
        for c in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                  "HeatingQC", "KitchenQual", "FireplaceQu",
                  "GarageQual", "GarageCond", "PoolQC"]:
            if c in af.columns:
                af[c] = af[c].map(qmap)
    num = af.dtypes[af.dtypes != "object"].index
    skewed = [c for c in num
              if af[c].skew() > 0.5 and af[c].dropna().min() >= 0]
    af[skewed] = np.log1p(af[skewed])
    af[num] = af[num].apply(lambda x: (x - x.mean()) / (x.std()))
    af[num] = af[num].fillna(0)
    af = pd.get_dummies(af, dummy_na=True, dtype=float)
    X = torch.tensor(af[:n_train].values, dtype=torch.float32)
    Xt = torch.tensor(af[n_train:].values, dtype=torch.float32)
    y = torch.log(torch.tensor(labels_raw))         # 本轮全部用对数标签
    return X, y, Xt


def get_net(in_features):
    return nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))


loss = nn.MSELoss()


def metric(net, Xb, yb):                            # 对数标签: 直接RMSE即指标
    with torch.no_grad():
        return torch.sqrt(torch.mean((net(Xb) - yb) ** 2)).item()


def train_es(net, Xtr, ytr, Xv, yv, lr, wd=0.01, bs=64,
             max_epochs=600, patience=60):
    it = d2l.load_array((Xtr, ytr), bs)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best_v, best_t, best_ep = float("inf"), float("inf"), 0
    no_improve = 0
    for ep in range(1, max_epochs + 1):
        for xb, yb in it:
            opt.zero_grad()
            loss(net(xb), yb).backward()
            opt.step()
        v = metric(net, Xv, yv)
        if v < best_v - 1e-4:
            best_v, best_ep = v, ep
            best_t = metric(net, Xtr, ytr)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_t, best_v, best_ep


def k_fold_es(X, y, lr, k=5):
    fs = X.shape[0] // k
    ts, vs, eps = [], [], []
    for i in range(k):
        torch.manual_seed(7)
        idx = slice(i * fs, (i + 1) * fs)
        mask = torch.ones(X.shape[0], dtype=torch.bool)
        mask[idx] = False
        t, v, ep = train_es(get_net(X.shape[1]), X[mask], y[mask],
                            X[idx], y[idx], lr)
        ts.append(t)
        vs.append(v)
        eps.append(ep)
    return float(np.mean(ts)), float(np.mean(vs)), int(np.mean(eps))


LEVELS = {1: "+log&skew", 2: "+construct", 3: "+ordinal"}
GRID_LR = [0.01, 0.03, 0.05, 0.1, 0.2]
BASELINE = 0.1317                                  # fe.py L0 的成绩

rows = []
for level, lname in LEVELS.items():
    X, y, Xt = build_features(level)
    for lr in GRID_LR:
        t, v, ep = k_fold_es(X, y, lr)
        rows.append(dict(level=level, name=lname, lr=lr, train=t, valid=v,
                         gap=v - t, best_epoch=ep))
        print(f"L{level} {lname:<11} lr={lr:<5} train={t:.4f} "
              f"valid={v:.4f} gap={v-t:+.4f} ep={ep}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "fe2_results.csv", index=False)
best = df.loc[df["valid"].idxmin()]
print(f"\nBEST: L{best.level} {best.name} lr={best.lr} valid={best.valid:.4f} "
      f"(baseline={BASELINE})")

if best.valid < BASELINE:                          # 只有优于基线才重新生成提交
    X, y, Xt = build_features(int(best.level))
    n_final = int(best.best_epoch)
    torch.manual_seed(7)
    net = get_net(X.shape[1])
    it = d2l.load_array((X, y), 64)
    opt = torch.optim.Adam(net.parameters(), lr=float(best.lr), weight_decay=0.01)
    for _ in range(n_final):
        for xb, yb in it:
            opt.zero_grad()
            loss(net(xb), yb).backward()
            opt.step()
    preds = np.exp(net(Xt).detach().numpy().reshape(-1))
    test_data["SalePrice"] = pd.Series(preds)
    pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1).to_csv(
        OUT / "submission.csv", index=False)
    print(f"final: level={int(best.level)} lr={best.lr} {n_final} epochs "
          f"-> submission.csv (预期Kaggle ~{best.valid:.4f})")
else:
    print("feature engineering did not beat baseline, submission.csv unchanged")
