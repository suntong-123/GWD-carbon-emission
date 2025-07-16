import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from matplotlib import gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import shap
# ==== 可视化配置 ====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==== 数据加载 ====
df = pd.read_csv(r"C:/Users/13600K/Desktop/diabetes_imputed7 - 5-1000（优化特征）.csv")
df.columns = df.columns.str.strip()
# 通过索引选择列范围（索引从0开始）
start_col_idx = 5  # 起始列索引（例如第3列）
end_col_idx = 17   # 结束列索引（例如第10列，不包含该列）
df_numeric = df.iloc[:, start_col_idx:end_col_idx]  # 选择指定列范围
print(df_numeric.head())

# 确保目标列存在
target_col = "SkinThickness"
assert target_col in df_numeric.columns, f"列 {target_col} 不存在，实际列名：{df_numeric.columns.tolist()}"


# ==== 模型配置 ====
model_config = {
    "n_estimators": 300,
    "min_samples_split": 10,
    "max_depth": None,
    "random_state": 42
}
model = RandomForestRegressor(**model_config)

# ==== 基准模型训练 ====
#def train_baseline_model():
X = df_numeric.drop(target_col, axis=1)
y = df_numeric[target_col]
print(X.head())
print(y.head())
    # 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练流程
model.fit(X_train, y_train)
print(np.shape(X_test))
    # 测试评估
y_pred = model.predict(X_test)

    # 评估计算
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
r2_val = r2_score(y_test, y_pred)

    # 控制台输出
print(f"RMSE: {rmse_val:.4f}")
print(f"R²: {r2_val:.4f}")
"""
    # 结果可视化
fig = plt.figure(figsize=(16, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3])

# 子图1：回归散点图
ax1 = plt.subplot(gs[0])
sns.regplot(x=y_test, y=y_pred,
                scatter_kws={"color": "steelblue", "alpha": 0.6, "edgecolor": "w"},
                line_kws={"color": "firebrick"}, ax=ax1)
ax1.text(0.05, 0.95, f"$R^2$ = {r2_val:.2f}", transform=ax1.transAxes,
             fontsize=14, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.set_title("真实值 vs 预测值", fontsize=14)
ax1.set_xlabel("真实值", fontsize=12)
ax1.set_ylabel("预测值", fontsize=12)

    # 子图2：特征重要性
ax2 = plt.subplot(gs[1])
importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

sns.barplot(x=importances.values, y=importances.index,
                palette="Blues_d", ax=ax2)
ax2.set_title("特征重要性排名", fontsize=14)
ax2.set_xlabel("重要性分数", fontsize=12)
ax2.set_ylabel("")
plt.tight_layout()
plt.show()

final_model = RandomForestRegressor(**model_config)

# 准备完整数据集
X_full = df_numeric.drop(target_col, axis=1)
y_full = df_numeric[target_col]

# 训练最终模型
final_model.fit(X_full, y_full)

# 新增模型保存环节
model_path = r"C:/Users\st\Desktop\final_model.joblib"  # Windows路径

# model_path = "./final_model.joblib"  # Mac/Linux相对路径
model_metadata = {
    'model': final_model,
    'feature_names': X_full.columns.tolist()
}
joblib.dump(model_metadata, model_path)

# 更新输出信息
print(f"\n最终模型已使用全部数据训练完成，并保存至：{model_path}")
print(f"使用的特征数量：{len(X_full.columns)} 个")
print(f"总训练样本量：{len(X_full)} 条")
"""
# ================== SHAP分析部分 ==================
# 初始化SHAP解释器
explainer = shap.TreeExplainer(model)  # 使用训练好的模型
shap_values = explainer.shap_values(X_test)
"""
# 全局特征重要性（特征数自适应处理）
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test,
                 feature_names=X.columns.tolist(),  # 使用真实特征名称
                 plot_type="bar",
                 show=False)
plt.title("基于SHAP值的特征重要性排序")
plt.tight_layout()
plt.show()

# SHAP值分布图
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test,
                 feature_names=X.columns.tolist(),
                 show=False)
plt.title("SHAP值分布摘要图")
plt.tight_layout()
plt.show()

# 单个样本分析（以第一个样本为例）
plt.figure(figsize=(12,4))
shap.force_plot(explainer.expected_value,
                shap_values[0],  # 样本索引
                X_test.iloc[0],
                feature_names=X.columns.tolist(),
                matplotlib=True,
                show=False)
plt.title("样本个体贡献分析")
plt.tight_layout()
plt.show()

# 特征交互分析（以最高重要性特征为例）
main_feature = X.columns[np.argmax(np.abs(shap_values).mean(axis=0))]  # 自动选择主特征

# 计算交互值
shap_interaction_values = explainer.shap_interaction_values(X_test)

# 绘制交互图
shap.dependence_plot(main_feature,
                     shap_values,
                     X_test,
                     interaction_index="auto",  # 自动选择交互特征
                     feature_names=X.columns.tolist())
plt.title(f"{main_feature}的特征交互分析")
plt.show()
features_to_plot = X.columns[:12]  # 假设选择前12个特征

# 绘制每个特征的SHAP依赖图
print("\n开始SHAP分析...")
explainer = shap.TreeExplainer(model)
print("SHAP解释器初始化完成。正在计算SHAP值...")
shap_values = explainer.shap_values(X_test)
print(f"SHAP值计算完成，形状为: {shap_values.shape}")

num_features_to_plot = min(11, X_test.shape[1])
features_to_plot = X_test.columns[:num_features_to_plot].tolist()

output_dir_shap_regression = os.path.expanduser("~/Desktop/SHAP回归依赖图_仅LOWESS_全框_Arial") # 修改输出目录名
os.makedirs(output_dir_shap_regression, exist_ok=True)
print(f"SHAP依赖图将保存到: {output_dir_shap_regression}")

# LOWESS参数
lowess_frac = 0.3
lowess_it = 1

print(f"将为以下特征绘制SHAP依赖图 (带LOWESS): {features_to_plot}")
print(f"LOWESS参数: frac={lowess_frac}, it={lowess_it}")

for feature_name in features_to_plot:
    print(f"\n正在处理特征: {feature_name} ...")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=500)
    fig.set_facecolor('white')

    shap.dependence_plot(
        feature_name,
        shap_values,
        X_test,
        interaction_index=None,
        ax=ax,
        show=False,
        alpha=0.6,
        dot_size=25
    )

    try:
        current_feature_col_idx_in_X = X_test.columns.get_loc(feature_name)
    except KeyError:
        print(f"  警告: 特征 '{feature_name}' 在 X_test.columns 中未找到，跳过此特征。")
        plt.close(fig)
        continue

    x_values_current_feat = X_test[feature_name].values
    y_values_current_feat_shap = shap_values[:, current_feature_col_idx_in_X]

    sort_indices = np.argsort(x_values_current_feat)
    x_sorted = x_values_current_feat[sort_indices]
    y_sorted_shap = y_values_current_feat_shap[sort_indices]

    min_points_for_lowess = max(5, int(lowess_frac * len(np.unique(x_sorted))) + 1) if lowess_it > 0 else 3
    lowess_plotted_successfully = False
    if len(np.unique(x_sorted)) < min_points_for_lowess:
        print(f"  警告: 特征 '{feature_name}' 的唯一数据点 ({len(np.unique(x_sorted))}) 过少，无法进行LOWESS拟合。")
    else:
        try:
            lowess_fitted_curve = lowess(y_sorted_shap, x_sorted, frac=lowess_frac, it=lowess_it)
            smoothed_x_coords = lowess_fitted_curve[:, 0]
            smoothed_y_coords = lowess_fitted_curve[:, 1]
            ax.plot(smoothed_x_coords, smoothed_y_coords, color='darkred', linewidth=2,
                    label=f'LOWESS拟合 (frac={lowess_frac}, it={lowess_it})', zorder=10)
            lowess_plotted_successfully = True
            print(f"  特征 '{feature_name}' 的LOWESS拟合曲线绘制完成。")
        except Exception as e_lowess_main:
            print(f"  错误: 特征 '{feature_name}' 的LOWESS拟合失败: {e_lowess_main}")

    ax.set_xlabel(f'{feature_name}', fontsize=24)
    ax.set_ylabel(f'SHAP Value for {feature_name}', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, linestyle='--', alpha=0.6)

    for spine_pos in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine_pos].set_visible(True)
        ax.spines[spine_pos].set_linewidth(1.2)
        ax.spines[spine_pos].set_color('black')


    file_save_path = os.path.join(output_dir_shap_regression, f"SHAP_dependence_{feature_name}_LOWESS_fullframe_Arial.png")
    try:
        fig.savefig(file_save_path, dpi=500, bbox_inches='tight')
        print(f"  图表已保存到: '{file_save_path}'")
    except Exception as e_save:
        print(f"  错误: 保存图表 '{file_save_path}' 失败: {e_save}")

    plt.show()
    plt.close(fig)
"""
 # ================== SHAP分析部分 ==================
# 新增的SHAP交互值计算（添加在这里）
shap_interaction_values = explainer.shap_interaction_values(X_test)  # 正确的方法名
shap_values_numpy = np.array(shap_values)  # 转换为numpy数组
interaction_matrix = shap_interaction_values  # 创建交互矩阵变量

import pandas as pd
import numpy as np
import os
def save_interaction_matrix_to_excel(interaction_matrix, feature_names, output_dir="Desktop"):
    # 确保 interaction_matrix 是 numpy 数组
    interaction_matrix = np.array(interaction_matrix)

    # 检查 interaction_matrix 是否为三维数组（假设是的，否则进行适当调整）
    if interaction_matrix.ndim == 3:
        # 通常 SHAP 交互值的第三个维度是类别数，而二分类问题中我们可能只关注正类（1）
        interaction_matrix = interaction_matrix[1]  # 索引 1 表示正类
    else:
        # 如果不是我们假设的格式，则直接使用
        pass

    # 转换为 DataFrame
    df_interaction = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)

    # 创建输出路径（默认为桌面）
    desktop_path = os.path.expanduser(f"~/Desktop/SHAP_Interaction_Matrix.xlsx")

    # 输出到 Excel
    df_interaction.to_excel(desktop_path)

    print(f"SHAP 交互矩阵已保存到桌面路径：{desktop_path}")


# 调用函数保存
save_interaction_matrix_to_excel(interaction_matrix, X.columns.tolist())