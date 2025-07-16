import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, gaussian_kde
from scipy.spatial.distance import jensenshannon
from tabpfn import TabPFNRegressor

# 设置中文字体（Windows系统常用字体示例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据加载预处理
file_path = r"E:\D盘\全球特征子刊支撑材料\machine learning\中国（优化12）.csv"
selected_cols = pd.read_csv(file_path, nrows=0).columns[5:]
df = pd.read_csv(file_path, usecols=selected_cols)
df.columns = df.columns.str.strip()
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

# 数据分割
target_col = "SkinThickness"
assert target_col in df_numeric.columns, f"目标列 {target_col} 不存在"
X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
model = TabPFNRegressor()
model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# 计算指标
# 训练集指标
train_mse = mean_squared_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)
train_corr, _ = pearsonr(y_train, y_pred_train)

# 测试集指标
test_mse = mean_squared_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)
test_corr, _ = pearsonr(y_test, y_pred_test)

# 补充可视化所需指标
final_rmse = np.sqrt(test_mse)
final_mae = mean_absolute_error(y_test, y_pred_test)
pearson_corr = test_corr

# 计算JSD（杰森-香农散度）
y_test_hist, _ = np.histogram(y_test, bins=20, density=True)
y_pred_hist, _ = np.histogram(y_pred_test, bins=20, density=True)
jsd = jensenshannon(y_test_hist, y_pred_hist)

# 输出结果
print(f'训练集 MSE: {train_mse:.4f}')
print(f'测试集 MSE: {test_mse:.4f}')
print(f'训练集 R² Score: {train_r2:.4f}')
print(f'测试集 R² Score: {test_r2:.4f}')
print(f'训练集 相关系数: {train_corr:.4f}')
print(f'测试集 相关系数: {test_corr:.4f}')

# 可视化：训练集与测试集对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, alpha=0.6, label='数据点')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='基准线')
plt.xlabel('实际值 (SkinThickness)')
plt.ylabel('预测值 (SkinThickness)')
plt.title(f'训练集 (R² = {train_r2:.4f})')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('实际值 (SkinThickness)')
plt.ylabel('预测值 (SkinThickness)')
plt.title(f'测试集 (R² = {test_r2:.4f})')

plt.tight_layout()
plt.show()

# 特征重要性（TabPFN可能不支持，需异常处理）
try:
    feature_importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, feature_importance)
    plt.xlabel('特征重要性')
    plt.title('TabPFN特征重要性分析')
    plt.tight_layout()
    plt.show()
except AttributeError:
    print("当前版本不支持特征重要性分析，请升级或更换模型")

# ========== 核密度-散点图 1 ==========
xy = np.vstack([y_test, y_pred_test])
kde = gaussian_kde(xy)
density = kde(xy)

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
sc = ax.scatter(
    x=y_test,
    y=y_pred_test,
    c=density,
    cmap='coolwarm',
    alpha=0.5,
    edgecolor='none'
)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

# 设置图表属性
ax.set_xlabel('实际值', fontsize=14, fontweight='bold')
ax.set_ylabel('预测值', fontsize=14, fontweight='bold')
ax.set_title('TabPFN回归分析结果', fontsize=16, fontweight='bold')
ax.grid(True)

# 添加文本框显示指标
metrics_text = (
    f"R² = {test_r2:.4f}\n"
    f"RMSE = {final_rmse:.4f}\n"
    f"MAE = {final_mae:.4f}\n"
    f"Pearson's r = {pearson_corr:.4f}\n"
    f"JSD = {jsd:.4f}"
)
ax.text(
    0.02, 0.98,
    metrics_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
)

plt.tight_layout()
plt.savefig(r"C:\Users\st\Desktop\干活\tabpfn_kde_plot_1.png", dpi=300, bbox_inches='tight')
plt.show()

# ========== 核密度-散点图 2 ==========
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
sc = ax.scatter(
    x=y_test,
    y=y_pred_test,
    c=density,
    cmap='coolwarm',
    alpha=0.5,
    edgecolor='none'
)
cbar = plt.colorbar(sc)
cbar.set_label('密度', fontsize=12)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax.set_xlabel('实际值', fontsize=14, fontweight='bold')
ax.set_ylabel('预测值', fontsize=14, fontweight='bold')
ax.set_title('TabPFN预测分布', fontsize=16, fontweight='bold')
ax.grid(True)

# 添加文本框
ax.text(
    0.02, 0.98,
    metrics_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
)

plt.tight_layout()
plt.savefig(r"C:\Users\st\Desktop\干活\tabpfn_kde_plot_2.png", dpi=300, bbox_inches='tight')
plt.show()

# ========== 核密度图 ==========
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
sns.kdeplot(
    x=y_test,
    y=y_pred_test,
    cmap='coolwarm',
    fill=True,
    ax=ax
)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

ax.set_xlabel('实际值', fontsize=14, fontweight='bold')
ax.set_ylabel('预测值', fontsize=14, fontweight='bold')
ax.set_title('TabPFN核密度分布', fontsize=16, fontweight='bold')
ax.grid(True)

# 添加文本框
ax.text(
    0.02, 0.98,
    metrics_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8)
)

plt.tight_layout()
plt.savefig(r"C:\Users\st\Desktop\干活\tabpfn_kde_plot_3.png", dpi=300, bbox_inches='tight')
plt.show()
