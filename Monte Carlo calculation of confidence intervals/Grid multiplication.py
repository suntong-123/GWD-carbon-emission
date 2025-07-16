import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

# -------------------------------
# 配置参数
# -------------------------------
n_samples = 10_000  # 建议样本量（平衡精度与速度）
confidence_level = 0.95  # 置信水平（95%）
chunk_size = 128  # 分块大小（根据内存调整）


# -------------------------------
# 辅助函数
# -------------------------------
def read_raster(file_path):
    """读取栅格元数据"""
    with rasterio.open(file_path) as src:
        meta = src.meta.copy()
    return meta


def generate_samples(mean, width):
    """生成单个栅格单元的样本（向量化）"""
    sigma = np.clip(width / (2 * 1.96), a_min=0.0, a_max=None)
    samples = np.random.normal(loc=mean, scale=sigma, size=(n_samples, *mean.shape))
    return samples


def monte_carlo_ab_product(a_mean_chunk, a_width_chunk,
                           b_mean_chunk, b_width_chunk):
    """向量化计算单个块的置信区间和均值乘积"""
    # 强制转换为float64
    a_mean_chunk = a_mean_chunk.astype(np.float64)
    b_mean_chunk = b_mean_chunk.astype(np.float64)
    a_width_chunk = a_width_chunk.astype(np.float64)
    b_width_chunk = b_width_chunk.astype(np.float64)

    # 计算均值乘积
    product_mean = a_mean_chunk * b_mean_chunk

    # 生成样本（向量化）
    a_samples = generate_samples(a_mean_chunk, a_width_chunk)
    b_samples = generate_samples(b_mean_chunk, b_width_chunk)

    # 计算乘积分布
    ab_samples = a_samples * b_samples

    # 计算置信区间分位数
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = 100 - lower_percentile
    lower = np.percentile(ab_samples, lower_percentile, axis=0)
    upper = np.percentile(ab_samples, upper_percentile, axis=0)
    ci_width = upper - lower

    return ci_width, product_mean


def read_chunk(file_path, window):
    """安全读取指定窗口的栅格数据（自动处理越界和无效值）"""
    with rasterio.open(file_path) as src:
        # 调整窗口以避免越界
        actual_window = Window.from_slices(*window.toslices())
        data = src.read(1, window=actual_window)
        # 替换无效值为 NaN
        data[data < -1e30] = np.nan
        return data


def process_rasters(input_paths, output_paths):
    """分块处理大型栅格数据（自动适配不同尺寸栅格）"""
    # 1. 获取所有栅格的元数据并计算最小公共区域
    meta_list = []
    min_rows = np.inf
    min_cols = np.inf

    for key in input_paths.values():
        with rasterio.open(key) as src:
            current_rows, current_cols = src.height, src.width
            meta_list.append(src.meta)
            if current_rows < min_rows:
                min_rows = current_rows
            if current_cols < min_cols:
                min_cols = current_cols

    print(f"检测到最小公共区域尺寸：{min_rows}行 × {min_cols}列")

    # 使用第一个栅格的元数据作为输出基准（需调整尺寸）
    meta = meta_list[0].copy()
    meta['height'] = min_rows
    meta['width'] = min_cols

    # 2. 预处理：检查所有输入栅格的数值范围
    for key in input_paths.values():
        with rasterio.open(key) as src:
            data = src.read(1)  # 读取整个栅格（仅首波段）
            nodata_values = src.nodatavals  # 获取NODATA值
            if nodata_values:
                nodata = nodata_values[0]
                if np.isnan(nodata):
                    invalid_mask = np.isnan(data)
                else:
                    invalid_mask = data == nodata
                print(f"栅格 {key} 的无效值数量：{np.sum(invalid_mask)}")
            else:
                print(f"栅格 {key} 未定义NODATA值！")

    # 初始化输出栅格
    with rasterio.open(output_paths['ci'], 'w', **meta) as dst_ci:
        with rasterio.open(output_paths['product'], 'w', **meta) as dst_product:
            # 遍历每个块
            for i in range(0, min_rows, chunk_size):
                for j in range(0, min_cols, chunk_size):
                    # 定义当前块的窗口
                    current_width = min(chunk_size, min_cols - j)
                    current_height = min(chunk_size, min_rows - i)
                    window = Window(j, i, current_width, current_height)

                    # 读取四个栅格的当前块数据
                    a_mean_chunk = read_chunk(input_paths['a_mean'], window)
                    a_width_chunk = read_chunk(input_paths['a_width'], window)
                    b_mean_chunk = read_chunk(input_paths['b_mean'], window)
                    b_width_chunk = read_chunk(input_paths['b_width'], window)

                    # 创建NODATA掩码（任一栅格无效则标记为无效）
                    mask = (np.isnan(a_mean_chunk) |
                            np.isnan(a_width_chunk) |
                            np.isnan(b_mean_chunk) |
                            np.isnan(b_width_chunk) |
                            (a_mean_chunk < -1e30) |
                            (a_width_chunk < -1e30) |
                            (b_mean_chunk < -1e30) |
                            (b_width_chunk < -1e30))

                    # 计算置信区间和均值乘积
                    ci_chunk, product_chunk = monte_carlo_ab_product(
                        a_mean_chunk, a_width_chunk,
                        b_mean_chunk, b_width_chunk
                    )

                    # 应用掩码（无效区域设为NaN）
                    ci_chunk[mask] = np.nan
                    product_chunk[mask] = np.nan

                    # 写入输出文件（确保形状匹配）
                    # 调整数据形状为 (1, height, width)
                    ci_data = ci_chunk[np.newaxis, :, :] if ci_chunk.ndim == 2 else ci_chunk
                    product_data = product_chunk[np.newaxis, :, :] if product_chunk.ndim == 2 else product_chunk

                    dst_ci.write(ci_data, window=window)
                    dst_product.write(product_data, window=window)

                    # 打印进度
                    num_invalid = np.sum(mask)
                    print(
                        f"Processed block: row {i}-{i + current_height}, col {j}-{j + current_width} | Invalid pixels: {num_invalid}")

    print("处理完成！")


# -------------------------------
# 示例运行（需替换路径）
# -------------------------------
input_paths = {
'a_mean': r"E:\D盘\地下水碳排放计算\置信区间计算\4个tif-yanmo-4.12\HCO3-mean.tif",
    'a_width':r"E:\D盘\地下水碳排放计算\置信区间计算\4个tif-yanmo-4.12\ci-yanmo.tif",
    'b_mean': r"E:\D盘\地下水碳排放计算\置信区间计算\4个tif-yanmo-4.12\global-trend.tif",
    'b_width':r"E:\D盘\地下水碳排放计算\置信区间计算\4个tif-yanmo-4.12\grace-ci.tif"
}

output_paths = {
    'ci': 'ab_confidence_width_cleaned.tif',
    'product': 'ab_product_mean_cleaned.tif'
}

# 执行处理
process_rasters(input_paths, output_paths)


def plot_results(ci_path, product_path):
    with rasterio.open(ci_path) as src_ci:
        ci = src_ci.read(1)
    with rasterio.open(product_path) as src_product:
        product = src_product.read(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(ci, cmap='viridis', interpolation='nearest')
    ax1.set_title('Confidence Interval Width')

    ax2.imshow(product, cmap='plasma', interpolation='nearest')
    ax2.set_title('Mean Product (A × B)')

    plt.tight_layout()
    plt.show()

# plot_results(output_paths['ci'], output_paths['product'])
