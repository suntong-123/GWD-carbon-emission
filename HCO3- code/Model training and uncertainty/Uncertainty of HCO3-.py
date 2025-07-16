import numpy as np
import rasterio
import glob
import os
from scipy.stats import norm  # 新增正态分布支持


def calculate_confidence_width(stack, confidence_level=95):
    """
    计算基于正态分布假设的参数置信区间宽度
    Args:
        stack: 3D numpy数组 (模型数量 × 高度 × 宽度)
        confidence_level: 置信水平 (默认95%)
    Returns:
        置信区间宽度数组 (单位：μg/L)
    """
    alpha = (100 - confidence_level) / 200.0  # 双尾检验
    z_score = norm.ppf(1 - alpha)  # 计算Z分位数
    std_dev = np.nanstd(stack, axis=0, ddof=1)  # 样本标准差
    return (2 * z_score * std_dev).astype(np.float32)


def process_uncertainty(input_dir, output_dir):
    """处理地下水碳酸盐浓度不确定性分析"""

    # 读取输入文件
    tif_files = sorted(
        glob.glob(os.path.join(input_dir, 'model_*.tif')),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    if not tif_files:
        raise FileNotFoundError(f"输入目录 {input_dir} 缺少预测结果文件")

    # 获取基准元数据
    with rasterio.open(tif_files[0]) as src:
        meta = src.meta.copy()
        meta.update({
            'count': 1,
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999.0,
            'compress': 'lzw'
        })
        height, width = src.shape

    # 内存预分配与数据加载
    stack = np.empty((len(tif_files), height, width), dtype=np.float32)
    for i, file_path in enumerate(tif_files):
        with rasterio.open(file_path) as src:
            if src.shape != (height, width):
                raise ValueError(f"文件 {file_path} 尺寸不匹配")
            stack[i, :, :] = src.read(1, masked=True).filled(np.nan)

    # 数值预处理（四舍五入到小数点后两位）
    stack = np.round(stack, decimals=2)

    # 定义通用指标计算函数
    def calc_metric(func, **kwargs):
        return func(stack, axis=0, **kwargs).astype(np.float32)

    # 计算关键指标
    metrics = {
        'mean': calc_metric(np.nanmean),  # 平均浓度
        'sd': calc_metric(np.nanstd),  # 标准差
        'cv': calc_metric(np.nanstd) / np.abs(calc_metric(np.nanmean)) * 100,  # 变异系数
        'ci_width': calculate_confidence_width(stack)  # 90%参数置信区间宽度
    }

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义栅格写入函数
    def write_geotiff(data_array, filename):
        """将数据数组写入GeoTIFF文件"""
        with rasterio.open(os.path.join(output_dir, filename), 'w', **meta) as dst:
            valid_data = np.where(np.isnan(data_array), meta['nodata'], data_array)
            dst.write(valid_data, 1)
            dst.set_band_description(1, filename[:-4])

    # 写入结果文件
    result_files = {
        'global_HCO3_map_mean.tif': metrics['mean'],
        'global_HCO3_map_sd.tif': metrics['sd'],
        'global_HCO3_map_cv.tif': metrics['cv'],
        'global_HCO3_map_CI_width.tif': metrics['ci_width']
    }

    for filename, data in result_files.items():
        write_geotiff(data, filename)

    # 质量控制输出
    ci_stats = (
        np.nanpercentile(metrics['ci_width'], [25, 50, 75]).tolist(),
        np.nanmean(metrics['ci_width']),
        np.nanstd(metrics['ci_width'])
    )
    print(f"✅ 处理完成 - 输出至：{os.path.abspath(output_dir)}")
    print(f"置信区间宽度统计（μg/L）:\n"
          f"中位数: {ci_stats[0][1]:.2f}\n"
          f"平均值: {ci_stats[1]:.2f} ± {ci_stats[2]:.2f}\n"
          f"四分位间距: {ci_stats[0][0]:.2f} - {ci_stats[0][2]:.2f}")


if __name__ == '__main__':
    # ====================
    # 用户配置区
    # ====================
    CONFIG = {
        "input_dir": r"E:\D盘\稳健性测试\模型\稳健性测试\多模型预测结果\tif总",  # 指向包含RF_pred文件的目录
        "output_dir": r"E:\D盘\稳健性测试\模型\稳健性测试\多模型预测结果"  # 新目录将自动创建
    }

    try:
        process_uncertainty(**{k: os.path.expanduser(v) for k, v in CONFIG.items()})
    except FileNotFoundError as fe:
        print(f"❌ 文件系统错误: {str(fe)}")
    except ValueError as ve:
        print(f"❌ 数据异常: {str(ve)}")
    except Exception as e:
        print(f"❌ 未捕获错误: {str(e)}")
