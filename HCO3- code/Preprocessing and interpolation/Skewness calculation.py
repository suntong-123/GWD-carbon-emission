import pandas as pd
import numpy as np
from scipy.stats import skew, gaussian_kde, norm
import matplotlib.pyplot as plt
import os

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'

# 设置全局字体大小（适用于标题、标签、图例等）
plt.rcParams['font.size'] = 14

# 设置坐标轴数字字体大小
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

file_path = r"E:\D盘\全球特征子刊支撑材料\machine learning\diabetes_imputed7 -（5-1000）（优化特征11）.csv"

try:
    df = pd.read_csv(file_path)
    skin_thickness = df["SkinThickness"].dropna()
    skewness = skew(skin_thickness)

    print(f"数据样本量: {len(skin_thickness)}")
    print(f"偏度(Skewness): {skewness:.4f}")

    plt.figure(figsize=(12, 5))

    # 直方图与分布拟合
    plt.subplot(1, 2, 1)
    # 绘制密度归一化的直方图
    n, bins, patches = plt.hist(skin_thickness, bins=30,
                                edgecolor='black', alpha=0.7,
                                density=True, label='Histogram')

    # 生成拟合数据范围
    x_min, x_max = skin_thickness.min(), skin_thickness.max()
    x = np.linspace(x_min, x_max, 500)

    # 拟合正态分布
    mu, sigma = norm.fit(skin_thickness)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')

    # 添加标签和标题
    plt.title('Bicarbonate Distribution')
    plt.xlabel('Bicarbonate')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    # 显示偏度
    plt.text(0.6, 0.6, f'Skewness: {skewness:.2f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # 保存或显示图表
    plt.tight_layout()

    # 获取桌面路径并保存图像
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    output_file = os.path.join(desktop_path, 'Bicarbonate_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"图表已保存到桌面: {output_file}")

    # 显示图表
    plt.show()

except FileNotFoundError:
    print("文件未找到，请检查路径是否正确。")
except Exception as e:
    print(f"发生错误: {e}")
