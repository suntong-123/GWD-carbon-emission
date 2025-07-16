import numpy as np
import rasterio
from tqdm import tqdm  # 进度条


def calculate_sum_with_ci_monte_carlo(
        mean_raster_path,
        total_width_raster_path,
        n_simulations=1000,
        confidence_level=0.95
):
    """
    使用蒙特卡洛模拟计算栅格总和及其置信区间

    参数:
    - mean_raster_path: 本值（均值）栅格路径
    - total_width_raster_path: 全区间宽度栅格路径
    - n_simulations: 模拟次数（默认1000次）
    - confidence_level: 置信水平（默认95%）

    返回:
    - total_sum: 所有栅格值的总和
    - ci_lower: 总和置信区间下限
    - ci_upper: 总和置信区间上限
    - ci_width: 总和的置信区间全宽度
    """
    # 1. 读取栅格数据
    with rasterio.open(mean_raster_path) as src:
        mean_values = src.read(1)
        nodata = src.nodata

    with rasterio.open(total_width_raster_path) as src:
        total_width = src.read(1)

    # 2. 创建有效数据掩码（排除NODATA）
    mask = (mean_values != nodata) & ~np.isnan(mean_values) & ~np.isnan(total_width)
    mean_valid = mean_values[mask]
    width_valid = total_width[mask]

    # 3. 计算总和的真实值
    total_sum = np.sum(mean_valid)

    # 4. 蒙特卡洛模拟
    simulated_sums = np.zeros(n_simulations)

    for i in tqdm(range(n_simulations), desc="Monte Carlo Simulation"):
        # 从每个像元的正态分布中采样
        sigma = width_valid / (2 * 1.96)  # 全宽度 -> 标准差
        simulated_values = np.random.normal(
            loc=mean_valid,
            scale=sigma
        )
        simulated_sums[i] = np.sum(simulated_values)

    # 5. 计算置信区间
    alpha = 1 - confidence_level
    ci_lower = np.percentile(simulated_sums, 100 * alpha / 2)
    ci_upper = np.percentile(simulated_sums, 100 * (1 - alpha / 2))
    ci_width = ci_upper - ci_lower

    return total_sum, ci_lower, ci_upper, ci_width


# 示例调用
total_sum, ci_lower, ci_upper, ci_width = calculate_sum_with_ci_monte_carlo(
    "mean_values.tif",
    "total_width.tif",
    n_simulations=1000
)

print(f"总和: {total_sum:.2f}")
print(f"置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"置信区间全宽度: {ci_width:.2f}")