import pandas as pd
import numpy as np
from tqdm import tqdm
from fancyimpute import KNN

input_file_path = r"D:\Results & Fruits\G3P Aquifers\G3P_Climate_and_Riverbasin_GWS.xlsx"
output_file_path = r"D:\Results & Fruits\G3P Aquifers\SARIMA\G3P_Climate_KNN.xlsx"
K_file_path = r"D:\Results & Fruits\G3P Aquifers\SARIMA\Climate_KNN.txt"

df = pd.read_excel(input_file_path)
best_k_values = {}
columns_with_missing = df.columns[df.isnull().any()].tolist()
columns_dict = {}

with open(K_file_path, 'w') as f:
    for col in tqdm(columns_with_missing, total=len(columns_with_missing)):
        col_index = df.columns.get_loc(col)
        min_diff = float('inf')
        best_k = 999

        cols_to_select = [df.columns[0], df.columns[col_index]]
        df_waiting_fill = df[cols_to_select]

        original_non_null = df[col].dropna()
        n = len(original_non_null)
        if n == 0:
            # 如果列中没有非缺失值，则跳过
            continue

        original_sum_of_squares = np.sum(original_non_null ** 2)
        original_mean_square_error = original_sum_of_squares / n

        for i in range(2, 11):
            knn_imputer = KNN(k=i)
            df_filled = knn_imputer.fit_transform(df_waiting_fill)
            df_filled = pd.DataFrame(df_filled, columns=cols_to_select)

            filled_non_null = df_filled[col].dropna()
            filled_sum_of_squares = np.sum(filled_non_null ** 2)
            filled_mean_square_error = filled_sum_of_squares / len(filled_non_null)
            diff = abs(original_mean_square_error - filled_mean_square_error)
            f.write(f"{col}, K: {i}, diff={diff:.4f}\n")

            if diff < min_diff:
                min_diff = diff
                best_k = i

        f.write(f"{col}, original_MSE={original_mean_square_error:.4f}, Min_diff: {min_diff:.4f}, K: {best_k}\n")
        knn_imputer = KNN(k=best_k)
        df_filled = knn_imputer.fit_transform(df_waiting_fill)
        df_filled = pd.DataFrame(df_filled, columns=cols_to_select)
        column_data = df_filled.iloc[:, 1]
        columns_dict[col] = column_data

df_filled_all = pd.DataFrame(columns_dict)
df_filled_all.to_excel(output_file_path, index=False)

print("Over")
