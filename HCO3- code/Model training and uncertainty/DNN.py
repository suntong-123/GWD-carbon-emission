import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import jensenshannon
import warnings

 #设置中文字体（Windows系统常用字体示例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
df = pd.read_csv(r"C:\Users\doomsday\Desktop\干活\2025.4.12\diabetes_imputed7 -（5-1000）（优化特征11）.csv")

# 处理列名，去除可能的空格
df.columns = df.columns.str.strip()

# 选择数值列
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

# 目标列
target_col = "SkinThickness"
assert target_col in df_numeric.columns, f"列 {target_col} 不在数据集中，实际列名：{df_numeric.columns.tolist()}"

print(f"""
=============================================
预测变量 ({target_col}) 方差分析:
+ 总体数据量: {len(df_numeric)}
+ 总体均值: {df_numeric[target_col].mean():.4f}
+ 总体方差 (σ², ddof=0): {np.var(df_numeric[target_col]):.4f}
=============================================
""")

# 定义 RMSE 计算函数
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# ========== 🚀 深度学习修改部分 ==========
# 特征归一化
scaler = StandardScaler()
X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col].values

X_scaled = scaler.fit_transform(X)

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 构建深度学习模型
def build_model(input_shape):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # 回归任务无激活函数
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    return model


# 早停和优化器回调
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

# 训练配置
model = build_model(X_train.shape[1])
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=1000,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# ========== 📊 训练过程可视化 ==========
plt.figure(figsize=(12, 4))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('Training and Validation Loss Curve')

# 绘制MAE曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('Training and Validation MAE Curve')

plt.tight_layout()
plt.savefig(
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\DNN\training_curves.png",
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ========== 🔍 模型评估 ==========
y_pred = model.predict(X_test).flatten()

final_rmse = rmse(y_test, y_pred)
final_mae = mae(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]

y_test_hist, _ = np.histogram(y_test, bins=20, density=True)
y_pred_hist, _ = np.histogram(y_pred, bins=20, density=True)
jsd = jensenshannon(y_test_hist, y_pred_hist)

print(f"最终测试集上的 RMSE: {final_rmse:.4f}")
print(f"最终测试集上的 R²: {r2:.4f}")
print(f"最终测试集上的 MAE: {final_mae:.4f}")
print(f"最终测试集上的 Pearson 相关系数: {pearson_corr:.4f}")
print(f"最终测试集上的 JSD: {jsd:.4f}")

# ========== 📈 回归可视化 ==========
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred,
            scatter_kws={"color": "blue", "alpha": 0.5},
            line_kws={"color": "red"})

plt.text(x=min(y_test), y=max(y_pred),
         s=f"$R^2$ = {r2:.4f}",
         fontsize=12,
         color="black",
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title("DNN Performance: True vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\DNN\DNN Performance.png",
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
ax.set_title('DNN Model', fontsize=18, fontweight='bold')

ax.grid(True)
ax.legend().set_visible(False)

test_metrics_text = (
    f"$R^2$:{r2:.4f}\n"
    f"RMSE:{final_rmse:.4f}\n"
    f"MAE:{final_mae:.4f}\n"
    f"Pearson's r:{pearson_corr:.4f}\n"
    f"JSD:{jsd:.4f}"
)

# 添加指标文本
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
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\DNN\DNN_kde_plot_1.png",  # 根据路径修改
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
ax.set_title('DNN Model', fontsize=18, fontweight='bold')

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
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\DNN\DNN_kde_plot_2.png",  # 根据路径修改
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
ax.set_title('DNN Model', fontsize=18, fontweight='bold')

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
    r"C:\Users\doomsday\Desktop\干活\2025.4.12\DNN\DNN_kde_plot_3.png",  # 根据路径修改
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()