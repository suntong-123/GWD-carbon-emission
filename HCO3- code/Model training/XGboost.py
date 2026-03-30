import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from scipy.spatial.distance import jensenshannon

#设置中文字体（Windows系统常用字体示例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
df = pd.read_csv(r"E:\D盘\全球特征子刊支撑材料\machine learning\diabetes_imputed7 -（5-1000）（优化特征11）.csv")

# 处理列名，去除可能的空格
df.columns = df.columns.str.strip()

# 选择数值列
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

# 目标列
target_col = "SkinThickness"
assert target_col in df_numeric.columns, f"列 {target_col} 不在数据集中，实际列名：{df_numeric.columns.tolist()}"

# 定义 RMSE 计算函数
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ========== 🔥 特征归一化 ==========
scaler = StandardScaler()
X = df_numeric.drop(target_col, axis=1)  # 特征
y = df_numeric[target_col]  # 目标变量

X_scaled = scaler.fit_transform(X)  # 归一化特征

# ========== 🔥 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.1, 
    max_depth=6,
    verbosity=0,
    objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = xgb_model.predict(X_test)

# 计算最终测试集 RMSE 和 R²
final_rmse = rmse(y_test, y_pred)
final_mae = mae(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]

y_test_hist, _ = np.histogram(y_test, bins=20, density=True)
y_pred_hist, _ = np.histogram(y_pred, bins=20, density=True)
jsd = jensenshannon(y_test_hist, y_pred_hist)

# print(f"5折交叉验证 RMSE: {mean_rmse:.4f}")
print(f"最终测试集上的 RMSE: {final_rmse:.4f}")
print(f"最终测试集上的 R²: {r2:.4f}")
print(f"最终测试集上的 MAE: {final_mae:.4f}")
print(f"最终测试集上的 Pearson 相关系数: {pearson_corr:.4f}")
print(f"最终测试集上的 JSD: {jsd:.4f}")

# ========== 📈 绘制真实值 vs 预测值的回归图，并标注 R² ==========
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={"color": "blue", "alpha": 0.5}, line_kws={"color": "red"})

# 标注 R² 值
plt.text(x=min(y_test), y=max(y_pred), s=f"$R^2$ = {r2:.4f}", fontsize=12, color="black",
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title(f"XGBoost Regression: True vs Predicted")
plt.grid(True)

plt.tight_layout()
plt.savefig(
    r"C:\Users\st\Desktop\干活\xgboost.png",
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ========== 📈 核密度-散点图 1==========
xy = np.vstack([y_test, y_pred])
kde = gaussian_kde(xy)
density = kde(xy)

fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

sc = plt.scatter(
    x=y_test,
    y=y_pred,
    c=density,
    cmap='coolwarm',
    alpha=0.5,
    edgecolor='none',
)

# 添加对角线参考线
x_min = min(y_test.min(), y_pred.min())
x_max = max(y_test.max(), y_pred.max())
ax.plot([x_min, x_max], [x_min, x_max], 'k--')

# 设置标签和标题
ax.set_xlabel('True Value', fontsize=16, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=16, fontweight='bold')
ax.set_title('XGBoost Model', fontsize=18, fontweight='bold')

ax.grid(True)
ax.legend().set_visible(False)

test_metrics_text = (
    f"$R^2$:{r2:.4f}\n"
    f"RMSE:{final_rmse:.4f}\n"
    f"MAE:{final_mae:.4f}\n"
    f"Pearson's r:{pearson_corr:.4f}\n"
    f"JSD:{jsd:.4f}"
)

# 添加R²指标文本
ax.text(
    0.05, 0.95,
    f"{test_metrics_text}",
    transform=ax.transAxes,
    fontsize=15,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left',
    color='black',
)

# 保存图像并显示
plt.savefig(
    r"C:\Users\st\Desktop\干活\xgboost_kde_plot_1.png",  # 根据路径修改
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ========== 📈 核密度-散点图 2==========
xy = np.vstack([y_test, y_pred])
kde = gaussian_kde(xy)
density = kde(xy)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

sc = plt.scatter(
    x=y_test,
    y=y_pred,
    c=density,
    cmap='coolwarm',
    alpha=0.5,
    edgecolor='none',
)
cbar = plt.colorbar(sc)
cbar.set_label('Density')

# 添加对角线参考线
x_min = min(y_test.min(), y_pred.min())
x_max = max(y_test.max(), y_pred.max())
ax.plot([x_min, x_max], [x_min, x_max], 'k--')

# 设置标签和标题
ax.set_xlabel('True Value', fontsize=16, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=16, fontweight='bold')
ax.set_title('XGBoost Model', fontsize=18, fontweight='bold')

ax.grid(True)
ax.legend().set_visible(False)

test_metrics_text = (
    f"$R^2$:{r2:.4f}\n"
    f"RMSE:{final_rmse:.4f}\n"
    f"MAE:{final_mae:.4f}\n"
    f"Pearson's r:{pearson_corr:.4f}\n"
    f"JSD:{jsd:.4f}"
)

# 添加R²指标文本
ax.text(
    0.05, 0.95,
    f"{test_metrics_text}",
    transform=ax.transAxes,
    fontsize=15,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left',
    color='black',
)
ax.tick_params(axis='both', which='major', labelsize=20)
# 保存图像并显示
plt.savefig(
    r"C:\Users\st\Desktop\干活\xgboost_kde_plot_2.png",  # 根据路径修改
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ========== 📈 核密度图 ==========
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

sns.kdeplot(
    x=y_test,
    y=y_pred,
    cmap='coolwarm',
    fill=True,
    ax=ax
)

# 添加对角线参考线
x_min = min(y_test.min(), y_pred.min())
x_max = max(y_test.max(), y_pred.max())
ax.plot([x_min, x_max], [x_min, x_max], 'k--')

# 设置标签和标题
ax.set_xlabel('True Value', fontsize=26, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=16, fontweight='bold')
ax.set_title('XGBoost Model', fontsize=18, fontweight='bold')

ax.grid(True)
ax.legend().set_visible(False)

test_metrics_text = (
    f"$R^2$:{r2:.4f}\n"
    f"RMSE:{final_rmse:.4f}\n"
    f"MAE:{final_mae:.4f}\n"
    f"Pearson's r:{pearson_corr:.4f}\n"
    f"JSD:{jsd:.4f}"
)

# 添加R²指标文本
ax.text(
    0.05, 0.95,
    f"{test_metrics_text}",
    transform=ax.transAxes,
    fontsize=15,
    fontweight='bold',
    verticalalignment='top',
    horizontalalignment='left',
    color='black',
)

# 保存图像并显示
plt.savefig(
    r"C:\Users\st\Desktop\干活\xgboost_kde_plot_3.png",  # 根据路径修改
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()
