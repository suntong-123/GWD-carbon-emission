import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import jensenshannon
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#建议对>10,000样本数据采用线性核
# 数据加载和预处理 ---------------------------------
file_path = r"C:\Users\doomsday\Desktop\干活\2025.4.12\diabetes_imputed7 -（5-1000）（优化特征11）.csv"

selected_cols = pd.read_csv(file_path, nrows=0).columns[5:]
df = pd.read_csv(file_path, usecols=selected_cols)

df.columns = df.columns.str.strip()
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

target_col = "SkinThickness"
assert target_col in df_numeric.columns, f"列 {target_col} 不存在，实际列名：{df_numeric.columns.tolist()}"

# 数据分割 --------------------------------------
X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM模型设置（包含标准化）--------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
svm_pipeline = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf')
)

# 参数网格设置（扩展了更多参数）
params = {
    'svr__C': [0.1, 1, 10, 100],  # 增加更多C参数
    'svr__epsilon': [0.01, 0.1, 0.2, 0.3],  # 增加epsilon选项
    'svr__gamma': ['scale', 'auto', 0.01, 0.1]  # 新增gamma参数探索
}

grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=params,
    cv=kf,  # 使用预定义的交叉验证方法
    scoring='neg_root_mean_squared_error',  # 设置评分指标为RMSE
    verbose=1  # 显示搜索过程
)

# 执行网格搜索
grid.fit(X_train, y_train)

# 获取最佳模型
best_model = grid.best_estimator_
print(f"最好参数组合：{grid.best_params_}")

# 验证结果评估
y_pred = best_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)
pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]

y_test_hist, _ = np.histogram(y_test, bins=20, density=True)
y_pred_hist, _ = np.histogram(y_pred, bins=20, density=True)
jsd = jensenshannon(y_test_hist, y_pred_hist)

# 输出详细评价指标
print("\n优化后模型性能：")
print(f"测试集RMSE: {final_rmse:.4f}")
print(f"测试集R²: {r2:.4f}")
print(f"测试集MAE: {final_mae:.4f}")
print(f"最终测试集上的 Pearson 相关系数: {pearson_corr:.4f}")
print(f"最终测试集上的 JSD: {jsd:.4f}")

# 回归可视化 ------------------------------------
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred,
           scatter_kws={"color": "blue", "alpha": 0.5},
           line_kws={"color": "red"})
plt.text(x=min(y_test), y=max(y_pred),
        s=f"$R^2$ = {r2:.4f}",
        fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black'))
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("SVM Regression: True vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\SVM\SVM Regression True vs Predicted.png",
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
ax.set_title('SVM Model', fontsize=18, fontweight='bold')

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
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\SVM\SVM_kde_plot_1.png",  # 根据路径修改
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
ax.set_title('SVM Model', fontsize=18, fontweight='bold')

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
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\SVM\SVM_kde_plot_2.png",  # 根据路径修改
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
    shade=True,
    ax=ax
)

# 添加对角线参考线
x_min = min(y_test.min(), y_pred.min())
x_max = max(y_test.max(), y_pred.max())
ax.plot([x_min, x_max], [x_min, x_max], 'k--')

# 设置标签和标题
ax.set_xlabel('True Value', fontsize=16, fontweight='bold')
ax.set_ylabel('Predicted Value', fontsize=16, fontweight='bold')
ax.set_title('SVM Model', fontsize=18, fontweight='bold')

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
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\SVM\SVM_kde_plot_3.png",  # 根据路径修改
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()