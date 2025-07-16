import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
from tqdm import tqdm

# ----------------------- 配置参数区 -----------------------
INPUT_PATH = r"D:\Results & Fruits\G3P Aquifers\SARIMA\效果评估\挑选测试.xlsx"  
OUTPUT_PATH = r"D:\Results & Fruits\G3P Aquifers\SARIMA\效果评估\挑选测试_SA.xlsx"  
PARAMETERS_PATH = r"D:\Results & Fruits\G3P Aquifers\SARIMA\效果评估\挑选测试_SA_PARA.xlsx"  
SEASONAL_PERIOD = 12  
TEST_SIZE = 12  


ARIMA_PARAMS = {
    'start_p': 0, 'max_p': 2,  # AR 阶数范围 [0,2]
    'd': None, 'max_d': 1,  # 差分阶数范围 [0,1]
    'start_q': 0, 'max_q': 2,  # MA 阶数范围 [0,2]
    'start_P': 0, 'max_P': 1,  # 季节性AR 阶数范围 [0,1]
    'D': None, 'max_D': 1,  # 季节性差分阶数范围 [0,1]
    'start_Q': 0, 'max_Q': 1,  # 季节性MA 阶数范围 [0,1]
    'seasonal': True,  # 启用季节性模型
    'm': SEASONAL_PERIOD,  # 季节周期
    'stepwise': True,  # 逐步搜索（加速）
    'n_jobs': 4,  # 并行核心数（4核）
    'error_action': 'ignore',  # 忽略异常组合
    'suppress_warnings': True  # 禁用警告
}


# ---------------------------------------------------------

def load_data(file_path):
    """加载并预处理数据"""
    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
    df = df.asfreq('MS').dropna()  # 补齐频率并删除缺失值
    df.index = pd.to_datetime(df.index)
    print(f"数据形状: {df.shape}, 频率: {df.index.freq}")
    return df


def optimize_and_forecast(series, params):
    """单列参数寻优与预测"""
    try:
        model = auto_arima(
            series,
            **params,
            trace=False  # 关闭详细输出
        )

        best_order = model.order
        best_seasonal_order = model.seasonal_order
        aic = model.aic()

        forecast = model.predict(n_periods=12)
        return {
            'series': series.name,
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'aic': aic,
            'forecast': forecast
        }
    except Exception as e:
        print(f"列 {series.name} 预测失败: {str(e)}")
        return {
            'series': series.name,
            'order': None,
            'seasonal_order': None,
            'aic': np.nan,
            'forecast': np.nan
        }


def main():
    df = load_data(INPUT_PATH)

    forecast_index = pd.date_range(
        start=df.index[-1] + pd.DateOffset(months=1),
        periods=12,
        freq='MS'
    )
    forecast_df = pd.DataFrame(index=forecast_index)

    results = Parallel(n_jobs=1, verbose=1)(
        delayed(optimize_and_forecast)(df[col], ARIMA_PARAMS)
        for col in df.columns
    )

    param_records = []
    for res in results:
        col = res['series']
        forecast_df[col] = res['forecast']

        param_records.append({
            '列名': col,
            'order_p': res['order'][0] if res['order'] else np.nan,
            'order_d': res['order'][1] if res['order'] else np.nan,
            'order_q': res['order'][2] if res['order'] else np.nan,
            'seasonal_P': res['seasonal_order'][0],
            'seasonal_D': res['seasonal_order'][1],
            'seasonal_Q': res['seasonal_order'][2],
            'AIC': res['aic']
        })

    combined_df = pd.concat([df, forecast_df])
    combined_df.to_excel(OUTPUT_PATH)
    print(f"预测结果已保存至 {OUTPUT_PATH}")

    param_df = pd.DataFrame(param_records)
    param_df.to_excel(PARAMETERS_PATH, index=False)
    print(f"参数记录已保存至 {PARAMETERS_PATH}")


if __name__ == "__main__":
    main()
