import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
from scipy.spatial.distance import jensenshannon
import os
from scipy.stats import gaussian_kde

# --------------------------
# 设置中文字体（Windows系统常用字体示例）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置参数
RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2


# --------------------------

def load_and_preprocess_data(file_path, target_col):
    """数据加载与预处理"""
    try:
        df = pd.read_csv(file_path, usecols=lambda col: col not in range(10))  # 从第六列开始读取
        df.columns = df.columns.str.strip().str.replace(' ', '_')

        if target_col not in df.columns:
            raise ValueError(f"目标列 {target_col} 不存在，可用列：{df.columns.tolist()}")

        numeric_cols = df.select_dtypes(include=np.number).columns
        df_numeric = df[numeric_cols]

        if not np.issubdtype(df_numeric[target_col].dtype, np.number):
            raise TypeError(f"目标列 {target_col} 必须是数值类型")

        return df_numeric
    except Exception as e:
        print(f"数据加载错误: {str(e)}")
        raise


def evaluate_model(model, X_test, y_test):
    """模型评估"""
    y_pred = model.predict(X_test)
    y_test_hist, _ = np.histogram(y_test, bins=20, density=True)
    y_pred_hist, _ = np.histogram(y_pred, bins=20, density=True)
    return {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'Pearson_corr':np.corrcoef(y_test, y_pred)[0, 1],
        'JSD':jensenshannon(y_test_hist, y_pred_hist)
    }


def plot_results(y_true, y_pred, metrics, title_suffix="", save_path=None):
    """可视化结果"""
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    sns.regplot(
        x=y_true, y=y_pred,
        scatter_kws={"color": "blue", "alpha": 0.5, "edgecolor": "w"},
        line_kws={"color": "red", "lw": 2},
        ci=95,
        ax=ax
    )

    annotation_str = f"""
    $R^2$: {metrics['R²']:.4f}
    """
    ax.text(0.05, 0.95, annotation_str, transform=ax.transAxes,
            verticalalignment='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', alpha=0.3)

    ax.set_xlabel("True Values", fontsize=12)
    ax.set_ylabel("Predicted Values", fontsize=12)
    ax.set_title(f"Validation Plot {title_suffix}", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, "validation_plot.png"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    xy = np.vstack([y_true, y_pred])
    kde = gaussian_kde(xy)
    density = kde(xy)

    sc = plt.scatter(
        x=y_true,
        y=y_pred,
        c=density,
        cmap='coolwarm',
        alpha=0.5,
        edgecolor='none',
    )
    cbar = plt.colorbar(sc)
    cbar.set_label('Density')

    # 添加对角线参考线
    x_min = min(y_true.min(), y_pred.min())
    x_max = max(y_true.max(), y_pred.max())
    ax.plot([x_min, x_max], [x_min, x_max], 'k--')

    # 设置标签和标题
    ax.set_xlabel('True Value', fontsize=16, fontweight='bold')
    ax.set_ylabel('Predicted Value', fontsize=16, fontweight='bold')
    ax.set_title('RF Model', fontsize=18, fontweight='bold')

    ax.grid(True)
    ax.legend().set_visible(False)

    test_metrics_text = (
        
        f"$R^2$: {metrics['R²']:.4f}\n"
        f"RMSE: {metrics['RMSE']:.4f}\n"
        f"MAE: {metrics['MAE']:.4f}\n"
        f"Pearson's r: {metrics['Pearson_corr']:.4f}\n"
        f"JSD: {metrics['JSD']:.4f}"
        
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
    if save_path:
        plt.savefig(os.path.join(save_path, "density_plot.png"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.show()


# --------------------------
# 执行主程序
# --------------------------
if __name__ == "__main__":
    try:
        file_path = r"C:\Users\doomsday\Desktop\干活\2025.4.12\diabetes_imputed7 -（5-1000）（优化特征11）RF版.csv" # 仅定义一次路径
        df = pd.read_csv(file_path, nrows=0)
        target_col = "SkinThickness"
        df = load_and_preprocess_data(file_path, target_col)

        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=500, max_depth=5,
                                            random_state=RANDOM_STATE, n_jobs=-1))
        ])

        param_dist = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 5, 10, 15],
            'model__min_samples_split': [2, 5, 10]
        }

        search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=20, cv=N_SPLITS,
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"Best params: {search.best_params_}")
        final_model = search.best_estimator_

        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            final_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error'
        )
        print(f"Cross-Validation RMSE: {np.mean(-cv_scores):.4f} (±{np.std(cv_scores):.4f})")

        test_metrics = evaluate_model(final_model, X_test, y_test)
        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")

        feature_importance = pd.Series(
            final_model.named_steps['model'].feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importance.head(14).plot(kind='barh')
        plt.title("Top 12 Feature Importances")
        plt.tight_layout()
        plt.savefig(
            r"C:\Users\doomsday\Desktop\干活\2025.4.12\RF\Top 12 Feature Importances.png",
            format='png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

        y_pred_test = final_model.predict(X_test)
        save_path = r"C:\Users\doomsday\Desktop\干活\2025.4.12\RF"
        plot_results(
            y_test, 
            y_pred_test, 
            test_metrics, 
            title_suffix="(Test Set)",
            save_path=save_path)

    except Exception as e:
        print(f"主程序运行错误: {str(e)}")
