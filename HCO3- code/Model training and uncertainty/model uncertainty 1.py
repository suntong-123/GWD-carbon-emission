import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_squared_error, r2_score

# ==== 路径设置 ====
MODEL_SAVE_DIR = r"E:\D盘\稳健性测试\模型\稳健性测试\模型带特征"
FIGURE_SAVE_DIR = r"E:\D盘\稳健性测试\模型\稳健性测试\模型带特征\图稳健性"
# ================

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载 & 预处理
file_path = r
selected_cols = pd.read_csv(file_path, nrows=0).columns[5:]
df = pd.read_csv(file_path, usecols=selected_cols)
df.columns = df.columns.str.strip()

target_col = "SkinThickness"
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col]

# 模型配置
model_params = {
    "n_estimators": 300,
    "min_samples_split": 10,
    "random_state": 42
}

for i in range(20):
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=i, shuffle=True
    )

    # 模型训练
    rf_model = RandomForestRegressor(**model_params)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # ==== 关键修改部分：保存元数据 ====
    model_metadata = {
        'model': rf_model,
        'feature_names': X.columns.tolist()  # 保存特征名称列表
    }
    # 保存模型及元数据
    model_path = os.path.join(MODEL_SAVE_DIR, f'model_{i + 1}.joblib')
    dump(model_metadata, model_path)
    # ============================

    # 可视化设置
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_test, y=y_pred,
                scatter_kws={"color": "blue", "alpha": 0.3},
                line_kws={"color": "red", "lw": 1.5})

    final_rmse = rmse(y_test, y_pred)
    final_mae = mae(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_corr = np.corrcoef(y_test, y_pred)[0, 1]

    y_test_hist, _ = np.histogram(y_test, bins=20, density=True)
    y_pred_hist, _ = np.histogram(y_pred, bins=20, density=True)
    jsd = jensenshannon(y_test_hist, y_pred_hist)

    plt.text(x=min(y_test), y=max(y_pred),
             s=f"$R^2$ = {r2:.4f}",
             fontsize=12,
             bbox=dict(facecolor='white', edgecolor='black'))

    plt.xlabel("True Value", fontsize=12)
    plt.ylabel("Predicted Value", fontsize=12)
    plt.title(f"The result of the {i+1}th random split", fontsize=14)
    plt.grid(True, alpha=0.3)

    # 保存图像
    fig_path = os.path.join(FIGURE_SAVE_DIR, f'图二轮{i + 1}.png')
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.close()

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
    ax.set_title('RF Model', fontsize=18, fontweight='bold')

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

    # 设置坐标轴刻度字体大小
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)  # 修改x轴刻度字体大小
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)  # 修改y轴刻度字体大小


    # 保存图像并显示
    density_fig_path = os.path.join(FIGURE_SAVE_DIR, f'density_plot_{i+1}.png')
    plt.savefig(density_fig_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图像

    # 进度输出
    print(f"[完成] 第 {i + 1}/20 次运行")
    print(f"模型保存至：{model_path}")
    print(f"特征数量：{len(model_metadata['feature_names'])}")
    print(f"图表保存至：{fig_path}\n")

print("所有训练任务完成！")
