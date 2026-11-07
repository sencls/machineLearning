# -*- coding: utf-8 -*-
"""特征工程实验: 网络固定(Linear-in,64-ReLU-Linear-64,1),分级叠加特征工程

Level 0: 基线(d2l原始预处理,原始房价标签) —— 对照组
Level 1: 分布修正   —— 标签取对数(对齐Kaggle指标) + 偏态数值特征log1p
Level 2: 特征构造   —— 总面积/总卫浴/房龄/翻新龄/质量x面积 + MSSubClass改类别
Level 3: 顺序编码   —— Ex/Gd/TA/Fa/Po质量等级映射为5..1(保留顺序信息)

协议: 5折交叉验证 + 早停(patience=30), 每级扫描lr取最优, wd=0.01
"""
import torch                      # Windows铁律: torch第一个导入
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


def build_features(level):
    """按级别构造特征,返回 (X_train, y, X_test, 是否对数标签)"""
    af = pd.concat((train_data.iloc[:, 1:-1].copy(), test_data.iloc[:, 1:].copy()))

    if level >= 2:                                   # ---- 特征构造 ----
        af["TotalSF"] = (af["TotalBsmtSF"].fillna(0) + af["1stFlrSF"].fillna(0)
                         + af["2ndFlrSF"].fillna(0))
        af["TotalBath"] = (af["FullBath"].fillna(0) + 0.5 * af["HalfBath"].fillna(0)
                           + af["BsmtFullBath"].fillna(0)
                           + 0.5 * af["BsmtHalfBath"].fillna(0))
        af["Age"] = af["YrSold"] - af["YearBuilt"]
        af["RemodAge"] = af["YrSold"] - af["YearRemodAdd"]
        af["QualSF"] = af["OverallQual"] * af["GrLivArea"]
        af["MSSubClass"] = af["MSSubClass"].astype(str)   # 名义码->类别,数值大小无意义

    if level >= 3:                                   # ---- 顺序编码 ----
        qmap = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
        for c in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                  "HeatingQC", "KitchenQual", "FireplaceQu",
                  "GarageQual", "GarageCond", "PoolQC"]:
            if c in af.columns:
                af[c] = af[c].map(qmap)              # NaN保持NaN,进数值管线统一填充

    num = af.dtypes[af.dtypes != "object"].index
    if level >= 1:                                   # ---- 偏态修正 ----
        skewed = [c for c in num
                  if af[c].skew() > 0.5 and af[c].dropna().min() >= 0]
        af[skewed] = np.log1p(af[skewed])

    af[num] = af[num].apply(lambda x: (x - x.mean()) / (x.std()))
    af[num] = af[num].fillna(0)
    af = pd.get_dummies(af, dummy_na=True, dtype=float)

    X = torch.tensor(af[:n_train].values, dtype=torch.float32)
    Xt = torch.tensor(af[n_train:].values, dtype=torch.float32)
    y = torch.log(torch.tensor(labels_raw)) if level >= 1 else torch.tensor(labels_raw)
    return X, y, Xt, level >= 1


def get_net(in_features):
    return nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))


def metric(net, Xb, yb, log_target):
    """Kaggle指标: 对数价格空间RMSE"""
    with torch.no_grad():
        p = net(Xb)
        if log_target:                               # 模型直接输出log价格
            return torch.sqrt(torch.mean((p - yb) ** 2)).item()
        c = torch.clamp(p, 1, float("inf"))          # 原始价格输出,防log(<=0)
        return torch.sqrt(torch.mean((torch.log(c) - torch.log(yb)) ** 2)).item()


loss = nn.MSELoss()


def train_es(net, Xtr, ytr, Xv, yv, log_t, lr, wd=0.01, bs=64,
             max_epochs=300, patience=30):
    it = d2l.load_array((Xtr, ytr), bs)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    best_v, best_t, best_ep = float("inf"), float("inf"), 0
    no_improve = 0
    for ep in range(1, max_epochs + 1):
        for xb, yb in it:
            opt.zero_grad()
            loss(net(xb), yb).backward()
            opt.step()
        v = metric(net, Xv, yv, log_t)
        if v < best_v - 1e-4:
            best_v, best_ep = v, ep
            best_t = metric(net, Xtr, ytr, log_t)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_t, best_v, best_ep


def k_fold_es(X, y, log_t, lr, k=5):
    fs = X.shape[0] // k
    ts, vs, eps = [], [], []
    for i in range(k):
        torch.manual_seed(7)
        idx = slice(i * fs, (i + 1) * fs)
        mask = torch.ones(X.shape[0], dtype=torch.bool)
        mask[idx] = False
        t, v, ep = train_es(get_net(X.shape[1]), X[mask], y[mask],
                            X[idx], y[idx], log_t, lr)
        ts.append(t)
        vs.append(v)
        eps.append(ep)
    return float(np.mean(ts)), float(np.mean(vs)), int(np.mean(eps))


# ---------- 分级 x lr 网格 ----------
LEVELS = {0: "baseline", 1: "+log&skew", 2: "+construct", 3: "+ordinal"}
GRID_LR = [0.05, 0.2, 0.5, 2, 4]
rows = []
for level, lname in LEVELS.items():
    X, y, Xt, log_t = build_features(level)
    for lr in GRID_LR:
        t, v, ep = k_fold_es(X, y, log_t, lr)
        rows.append(dict(level=level, name=lname, lr=lr, train=t, valid=v,
                         gap=v - t, best_epoch=ep))
        print(f"L{level} {lname:<11} lr={lr:<5} train={t:.4f} "
              f"valid={v:.4f} gap={v-t:+.4f} ep={ep}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "fe_results.csv", index=False)

print("\n== 每级最优 ==")
best_per_level = df.loc[df.groupby("level")["valid"].idxmin()]
for _, r in best_per_level.iterrows():
    print(f"L{r.level} {r.name:<11} best lr={r.lr:<5} valid={r.valid:.4f} "
          f"(train={r.train:.4f}, ep={r.best_epoch})")

best = df.loc[df["valid"].idxmin()]
print(f"\nGLOBAL BEST: L{best.level} {best.name} lr={best.lr} valid={best.valid:.4f}")

# ---------- 用全局最优配置全量训练并生成提交 ----------
X, y, Xt, log_t = build_features(int(best.level))
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

preds = net(Xt).detach().numpy().reshape(-1)
if log_t:
    preds = np.exp(preds)                            # 模型输出log价格,还原
test_data["SalePrice"] = pd.Series(preds)
pd.concat([test_data["Id"], test_data["SalePrice"]], axis=1).to_csv(
    OUT / "submission.csv", index=False)
print(f"final: level={int(best.level)} lr={best.lr} {n_final} epochs "
      f"-> submission.csv (预期Kaggle ~{best.valid:.4f})")
