import numpy as np
import rasterio
from pymannkendall import original_test
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm


def read_raster(file_path):
    with rasterio.open(file_path) as src:
        return src.read()


def write_raster(output_path, data, meta):
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=meta['height'],
        width=meta['width'],
        count=1,
        dtype=data.dtype,
        crs=meta['crs'],
        transform=meta['transform'],
    ) as dst:
        dst.write(data, 1)


def process_row(i, data, confidence_level):
    num_months, rows, cols = data.shape
    row_results = np.zeros(cols, dtype=np.float32)
    z = norm.ppf(1 - (1 - confidence_level)/2)

    for j in range(cols):
        values = data[:, i, j]

        if (np.all(values == 0) or
            np.all(np.isnan(values)) or
            len(np.unique(values)) < 2):
            row_results[j] = np.nan
            continue

        n = len(values)
        if n < 2:
            row_results[j] = np.nan
            continue

        indices_i, indices_j = np.triu_indices(n, 1)
        delta = values[indices_j] - values[indices_i]
        denominator = indices_j - indices_i
        slopes = delta / denominator

        sen_slope = np.median(slopes)
        s_squared = np.var(slopes, ddof=1)
        k = n * (n - 1) // 2
        se = np.sqrt(s_squared / k)
        ci_range = 2 * z * se * 12
        row_results[j] = ci_range

    return row_results


def sen_ci_map_approximate(data, confidence_level=0.95):
    num_months, rows, cols = data.shape
    ci_range_map = np.zeros((rows, cols), dtype=np.float32)

    row_results_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(process_row)(i, data, confidence_level)
        for i in tqdm(range(rows), desc="计算置信区间", unit="行")
    )

    for i in range(rows):
        ci_range_map[i] = row_results_list[i]

    return ci_range_map


input_path = r"D:\Results & Fruits\G3P_Pixel\Results\G3P_GWS_Pixel_0.5_Full_Mask.tif"
ci_range_path = r"D:\Results & Fruits\G3P_Pixel\Results\G3P_GWS_Pixel_0.5_SenSlope_95%CI.tif"
data_stack = read_raster(input_path)
with rasterio.open(input_path) as src:
    meta = src.meta

confidence_level = 0.95
ci_range_map = sen_ci_map_approximate(data_stack, confidence_level)

print("min:", np.nanmin(ci_range_map))
print("max:", np.nanmax(ci_range_map))


meta.update(dtype=ci_range_map.dtype)
if ci_range_map.ndim == 3:
    ci_range_map = ci_range_map[0]

write_raster(ci_range_path, ci_range_map, meta)
print(f"计算完成，结果保存为: {ci_range_path}")
