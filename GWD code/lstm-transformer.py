import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import os
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

set_seed(42)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, dates, seq_length, pred_length):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.data = data
        self.dates = dates

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length:index + self.seq_length + self.pred_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class MultiScaleLSTMTransformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, fusion_size=128, transformer_d=128,
                 nhead=4, transformer_layers=2, dropout=0.2, pred_length=12):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm4 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, fusion_size),
            nn.ReLU(),
            nn.Linear(fusion_size, transformer_d))
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d, nhead=nhead, dropout=dropout,
                                                  batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        self.out_proj = nn.Linear(transformer_d, pred_length)

    def forward(self, x):
        device = x.device
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        x2 = x[:, ::2, :]
        x4 = x[:, ::4, :]
        _, (h1, _) = self.lstm1(x)
        _, (h2, _) = self.lstm2(x2)
        _, (h4, _) = self.lstm4(x4)
        fused = torch.cat([h1[-1], h2[-1], h4[-1]], dim=1)
        proj = self.fusion(fused)
        transformer_in = proj.unsqueeze(1).repeat(1, x.size(1), 1)
        transform_out = self.transformer(transformer_in)
        final_state = transform_out[:, -1, :]
        prediction = self.out_proj(final_state)
        return prediction, transform_out

def train_model(model, dataloader, criterion, optimizer, epochs, device='cpu'):
    model.to(device)
    loss_history = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            preds, _ = model(inputs)
            loss = criterion(preds.view(-1), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.5f}")
    return loss_history

@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, mean_val, std_val):
    model.eval()
    total_loss = 0.0
    y_true_list = []
    y_pred_list = []
    tfm_outs = []
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds, tfm = model(inputs)

        batch_true = targets.view(-1).cpu().numpy()
        batch_pred = preds.view(-1).cpu().numpy()
        batch_true_de = batch_true * std_val + mean_val
        batch_pred_de = batch_pred * std_val + mean_val

        y_true_list.extend(batch_true_de)
        y_pred_list.extend(batch_pred_de)
        tfm_outs.append(tfm.cpu().numpy())

        loss = criterion(preds.view(-1), targets.view(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
    r_squared = r2_score(y_true_list, y_pred_list)
    mae = mean_absolute_error(y_true_list, y_pred_list)

    return avg_loss, y_true_list, y_pred_list, tfm_outs, rmse, r_squared, mae

def load_and_preprocess(file_path, start_date_str='2000-01-01'):
    df = pd.read_csv(file_path)
    if 'Date' not in df.columns or 'GWL' not in df.columns:
        print(f"[ERROR] 文件 {file_path} 缺少必要列")
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Date'] >= start_date_str].reset_index(drop=True)
    if df.empty:
        return None
    data = df['GWL'].values
    dates = df['Date'].values
    mean_val = data.mean()
    std_val = data.std() + 1e-8
    normalized = (data - mean_val)/std_val
    return df, normalized, mean_val, std_val, dates

def plot_all(file_name, df_history, loss_curve, transformer_matrix, output_dir,
            y_test_true, y_test_pred, min_rmse, dates, seq_length):
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    # 图1：原始时间序列示例
    axs[0, 0].plot(df_history['Date'], df_history['GWL'], 'b-', alpha=0.6)
    axs[0, 0].set_title("Original Time Series Example")
    axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axs[0, 0].tick_params(axis='x', rotation=45)

    # 图2：训练损失曲线
    epochs = list(range(1, len(loss_curve)+1))
    axs[0, 1].plot(epochs, loss_curve, 'r-', marker='o')
    axs[0, 1].set_title("Training Loss Curve")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("MSE Loss")

    # 图3：Transformer注意力热图
    if transformer_matrix.size > 0:
        im = axs[1, 0].imshow(transformer_matrix[-416:].T, aspect='auto', cmap='viridis')
        axs[1, 0].set_title("Transformer Global Feature Heatmap")
        fig.colorbar(im, ax=axs[1, 0])
    else:
        axs[1, 0].text(0.5, 0.5, 'Transform输出异常', ha='center', va='center')

    # 图4：测试集预测对比
    num_samples = len(y_test_true)
    x_values = np.arange(num_samples)
    axs[1, 1].plot(x_values, y_test_true, 
                  marker='o',  # 添加圆形标记
                  markersize=5, 
                  linestyle='-', 
                  linewidth=1.5,
                  label='True Value',
                  alpha=0.8)
    axs[1, 1].plot(x_values, y_test_pred,  
                  marker='s',  # 添加方形标记
                  markersize=5,
                  linestyle='--',
                  linewidth=1.5,
                  label='Predicted Value')
    axs[1, 1].set_title(f"Predicted vs. Ground Truth (RMSE={min_rmse:.2f}m)", fontsize=11)
    axs[1, 1].set_xlabel("Sample Index")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].grid(linestyle='--', alpha=0.5)
    axs[1, 1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"Analysis_{file_name}.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 新增：单独保存图4到独立文件
    fig_chart4 = plt.figure(figsize=(12, 6))
    ax_chart4 = fig_chart4.add_subplot(111)
    
    ax_chart4.plot(x_values, y_test_true, 
                 marker='o',
                 markersize=5,
                 linestyle='-',
                 linewidth=1.5,
                 label='True Value',
                 alpha=0.8)
    ax_chart4.plot(x_values, y_test_pred, 
                 marker='s',
                 markersize=5,
                 linestyle='--',
                 linewidth=1.5,
                 label='Predicted Value')
    ax_chart4.set_title(f"Predicted vs. Ground Truth (RMSE={min_rmse:.2f}m)", fontsize=11)
    
    ax_chart4.set_xlabel("Sample Index")
    ax_chart4.set_ylabel("Value")
    ax_chart4.grid(linestyle='--', alpha=0.5)
    ax_chart4.legend()
    
    # 自动调整日期格式和布局
    
    # plt.tight_layout()
    
    # 保存独立图表
    output_filename = os.path.join(output_dir, f"预测对比_{file_name}_独立图.png")
    fig_chart4.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig_chart4)  # 及时释放内存

def plot_regression_analysis(file_name, output_dir, y_train_true, y_train_pred, y_test_true, y_test_pred, train_rmse, train_r2, test_rmse, test_r2, train_mae, test_mae):
    """
    绘制预测值与真实值的回归分析图
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=1200)
    ax.set_facecolor('#F0F0F0')  # 设置背景色
    
    # 绘制训练集散点
    ax.scatter(y_train_true, y_train_pred,
               color='#6A9ACE',
               edgecolor='#6A9ACE',
               s=20,
               alpha=0.6,
               label='Training Set',
               facecolors='none')
    
    # 绘制测试集散点
    ax.scatter(y_test_true, y_test_pred,
               color='#F1766D',
               edgecolor='#F1766D',
               s=20,
               alpha=0.6,
               label='Testing Set',
               facecolors='none')
    
    min_val = min(min(y_train_true), min(y_train_pred), min(y_test_true), min(y_test_pred))
    max_val = max(max(y_train_true), max(y_train_pred), max(y_test_true), max(y_test_pred))
    data_range = max_val - min_val
    padding = data_range * 0.05  # 可调整的扩展比例

    lims = [min_val - padding, max_val + padding]

    # 绘制参考线x=y
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    
    # 计算回归线
    # 训练集回归线
    m_train, b_train = np.polyfit(y_train_true, y_train_pred, 1)
    ax.plot(lims, m_train*np.array(lims)+b_train,
            color='#6A9ACE',
            linestyle='--',
            label=f'训练集回归线 (y={m_train:.2f}x+{b_train:.2f})')
    
    # 测试集回归线
    m_test, b_test = np.polyfit(y_test_true, y_test_pred, 1)
    ax.plot(lims, m_test*np.array(lims)+b_test,
            color='#F1766D',
            linestyle='--',
            label=f'测试集回归线 (y={m_test:.2f}x+{b_test:.2f})')
    
    # 设置坐标轴和标题
    ax.set_xlabel('True Value')
    ax.set_ylabel('Predicted Value')
    # ax.set_title('预测值与真实值的回归分析')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # 添加图例
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 添加指标文本
    # 左上角训练集指标
    train_text = f"Train Metrics：\nRMSE={train_rmse:.3f}\nMAE={train_mae:.3f}\nR2={train_r2:.3f}\n斜率={m_train:.2f}"
    ax.text(0.02, 0.98,
            train_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=None)
    
    # 右下角测试集指标
    test_text = f"Test Metrics：\nRMSE={test_rmse:.3f}\nMAE={test_mae:.3f}\nR2={test_r2:.3f}\n斜率={m_test:.2f}"
    ax.text(0.98, 0.02,
            test_text,
            transform=ax.transAxes,
            verticalalignment='bottom',
            ha='right',
            bbox=None)
    
    # 保存图表
    output_path = os.path.join(output_dir, f"回归分析_{file_name}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    plt.close()
    
def main():
    root_path = r"C:\Users\doomsday\Desktop\干活\代码+运行\lstm+transformer\逐周&月数据"
    output_base = r"C:\Users\doomsday\Desktop\干活\代码+运行\lstm+transformer\lstm_transformer"
    frequency_map = {
        "逐月数据": {"freq_key": "month", "seq_len": 12, "pred_length": 1},
        "逐周数据": {"freq_key": "week", "seq_len": 48, "pred_length": 1}
    }

    target_folder = "逐周数据"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params = frequency_map[target_folder]
    seq_length = params["seq_len"]
    pred_length = params["pred_length"]
    data_freq = params["freq_key"]
    
    folder_abspath = os.path.join(root_path, target_folder)
    os.makedirs(output_base, exist_ok=True)
    save_subdir = os.path.join(output_base, target_folder)
    os.makedirs(save_subdir, exist_ok=True)
    
    for file_name in os.listdir(folder_abspath):
        if not file_name.endswith('.csv'):
            continue
            
        file_path = os.path.join(folder_abspath, file_name)
        data_bundle = load_and_preprocess(file_path)
        
        if not data_bundle:
            print(f"跳过异常文件：{file_name}")
            continue
            
        df, norm_data, mean_val, std_val, dates = data_bundle
        dataset = TimeSeriesDataset(norm_data, dates, seq_length, pred_length)
        
        if len(dataset) < 1:
            print(f"文件 '{file_name}' 数据不足，跳过")
            continue
            
        total_len = len(dataset)
        train_size = int(0.8 * total_len)
        val_indices = list(range(train_size, total_len))
        print(f"【数据集统计】文件：{file_name}")
        print(f"总样本数：{total_len}")
        print(f"训练集大小：{train_size} ({train_size/total_len:.0%})")
        print(f"测试集大小：{len(val_indices)} ({len(val_indices)/total_len:.0%})")
        print("-"*40)
        
        # 正确创建子集 → 解决报错的核心修改
        train_set = Subset(dataset, range(train_size))
        test_set_full = Subset(dataset, val_indices)

        # ==== 添加随机搜索逻辑开始 ====
    
        # 定义超参数搜索空间
        param_space = {
            'hidden_dim': [32, 64, 128, 256],
            'num_layers': [1, 2, 3],
            'fusion_size': [64, 128, 256],
            'transformer_d': [64, 128, 256],
            'nhead': [2, 4, 8],
            'transformer_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'lr': [1e-5, 1e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100, 150]
        }
    
        num_searches = 10  # 随机采样次数（可调整）
        best_loss = float('inf')
        best_params = None
        best_model_path = None
        saved_models = []
    
        for search_idx in range(num_searches):
            # 重置随机种子以确保每次采样公平
            set_seed(42 + search_idx)
        
            # 随机采样参数
            params = {k: random.choice(v) for k, v in param_space.items()}
        
            # 确保nhead能整除transformer_d
            while params['transformer_d'] % params['nhead'] != 0:
                params['nhead'] = random.choice(param_space['nhead'])
        
            # 创建模型
            model = MultiScaleLSTMTransformer(
                input_dim=1,
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                fusion_size=params['fusion_size'],
                transformer_d=params['transformer_d'],
                nhead=params['nhead'],
                transformer_layers=params['transformer_layers'],
                dropout=params['dropout'],
                pred_length=pred_length
            ).to(device)
        
            # 初始化优化器
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
            criterion = nn.MSELoss()
        
            # 创建数据加载器
            train_loader = DataLoader(
                train_set, 
                batch_size=params['batch_size'], 
                shuffle=True
            )
        
            # 训练模型
            loss_history = train_model(
                model, 
                train_loader, 
                criterion, 
                optimizer, 
                epochs=params['epochs'],
                device=device
            )

            # 评估训练集
            train_loader_eval = DataLoader(train_set, batch_size=params['batch_size'], shuffle=False)
            train_loss, y_train_true, y_train_pred, _, train_rmse, train_r2, train_mae = evaluate_model(
                model, 
                train_loader_eval, 
                criterion, 
                device,
                mean_val,
                std_val
            )

            # 评估模型
            test_loader = DataLoader(test_set_full, batch_size=1)
            test_loss, y_test_true, y_test_pred, tfm_outs, test_rmse, test_r2, test_mae = evaluate_model(
                model, 
                test_loader, 
                criterion, 
                device,
                mean_val,
                std_val
            )

            # 保存完整评估结果
            saved_models.append({
                'params': params,
                'model': model.state_dict(),
                'loss': test_loss,
                'loss_curve': loss_history,
                'y_train_true': y_train_true,
                'y_train_pred': y_train_pred,
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'train_mae': train_mae,
                'y_test_true': y_test_true,
                'y_test_pred': y_test_pred,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'tfm_outs': tfm_outs
            })

            if test_loss < best_loss:
                best_loss = test_loss
            
        # 选择最优模型
        best_model_data = min(saved_models, key=lambda x: x['loss'])
        best_model = MultiScaleLSTMTransformer(
            input_dim=1,
            hidden_dim=best_model_data['params']['hidden_dim'],
            num_layers=best_model_data['params']['num_layers'],
            fusion_size=best_model_data['params']['fusion_size'],
            transformer_d=best_model_data['params']['transformer_d'],
            nhead=best_model_data['params']['nhead'],
            transformer_layers=best_model_data['params']['transformer_layers'],
            dropout=best_model_data['params']['dropout'],
            pred_length=pred_length
        ).to(device)
        best_model.load_state_dict(best_model_data['model'])

        # 直接使用最优模型的评估结果
        loss_curve = best_model_data['loss_curve']
        y_train_true = best_model_data['y_train_true']
        y_train_pred = best_model_data['y_train_pred']
        train_rmse = best_model_data['train_rmse']
        train_r2 = best_model_data['train_r2']
        train_mae = best_model_data['train_mae']
        y_test_true = best_model_data['y_test_true']
        y_test_pred = best_model_data['y_test_pred']
        test_rmse = best_model_data['test_rmse']
        test_r2 = best_model_data['test_r2']
        test_mae = best_model_data['test_mae']
        tfm_outs = best_model_data['tfm_outs']

        # 绘图
        plot_all(
            file_name=file_name,
            df_history=df,
            loss_curve=loss_curve,
            transformer_matrix=tfm_outs[0] if tfm_outs else np.zeros(1),
            output_dir=save_subdir,
            y_test_true=y_test_true,
            y_test_pred=y_test_pred,
            min_rmse=test_rmse,
            dates=dates,
            seq_length=seq_length
        )

        plot_regression_analysis(
            file_name=file_name,
            output_dir=save_subdir,
            y_train_true=y_train_true,
            y_train_pred=y_train_pred,
            y_test_true=y_test_true,
            y_test_pred=y_test_pred,
            train_rmse=train_rmse,
            train_r2=train_r2,
            train_mae=train_mae,
            test_rmse=test_rmse,
            test_r2=test_r2,
            test_mae=test_mae
        )
        
        test_data_rows = []

        for idx in range(len(test_set_full)):
            origin_index = test_set_full.indices[idx]
            pred_date = dates[origin_index + seq_length]
            true_val = y_test_true[idx]
            pred_val = y_test_pred[idx]
            test_data_rows.append({
                '样本序号': idx,
                '日期': pred_date,
                '真实值': true_val,
                '预测值': pred_val
            })

        test_df = pd.DataFrame(test_data_rows)
        output_filename = f"{os.path.splitext(file_name)[0]}_测试数据.xlsx"
        test_df.to_excel(os.path.join(save_subdir, output_filename), index=False)
        print(f"测试数据保存成功: {output_filename}")

        # 保存参数到文本文件
        param_log_path = os.path.join(save_subdir, f"参数记录_{file_name}.txt")
        with open(param_log_path, 'w', encoding='utf-8') as f:
            f.write(f"文件名：{file_name}\n")
            f.write(f"时间分辨率：{data_freq}\n")
            f.write("最佳参数组合：\n")
            for key, value in best_model_data['params'].items():
                f.write(f"  {key}: {value}\n")
            f.write(f"对应测试Loss：{best_loss: .4f}\n")

        # ==== 更新模型性能记录 ====
        model_performance = {
            '文件名': file_name,
            '时间分辨率': data_freq,
            '训练集Loss': round(train_loss, 4),
            '训练集RMSE': round(train_rmse, 4),
            '训练集R平方': round(train_r2, 4),
            '训练集MAE': round(train_mae, 4),
            '测试集Loss': round(test_loss, 4),
            '测试集RMSE': round(test_rmse, 4),
            '测试集R平方': round(test_r2, 4),
            '测试集MAE': round(test_mae, 4),
            '参数量': sum(p.numel() for p in model.parameters()),
            '预测步数': pred_length
        }

        # 将结果追加到汇总数据框
        if 'folder_performance' in locals():
            folder_performance.append(model_performance)
        else:
            folder_performance = [model_performance]
    
    if 'folder_performance' in locals():
        summary_df = pd.DataFrame(folder_performance)
        # 手动指定列的顺序（可选）
        summary_df = summary_df[[
            '文件名', 
            '时间分辨率',
            '训练集Loss', '训练集RMSE', '训练集R平方','训练集MAE',
            '测试集Loss', '测试集RMSE', '测试集R平方','测试集MAE',
            '参数量', '预测步数'
        ]]
        summary_path = os.path.join(output_base, f"{target_folder}_算法对比报告.xlsx")
        summary_df.to_excel(summary_path, index=False)
        print(f"算法对比报告已保存至：{summary_path}")

if __name__ == "__main__":
    main()
