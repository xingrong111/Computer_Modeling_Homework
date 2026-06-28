"""
可视化模块
生成所有论文用图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 设置中文字体（解决中文显示问题）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_data_overview():
    """图1：数据概览"""
    df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv', parse_dates=['date'])

    plt.figure(figsize=(14, 5))
    plt.plot(df['date'], df['new_confirmed'], alpha=0.4, label='原始每日新增')
    plt.plot(df['date'], df['new_confirmed_smooth'], 'r-', linewidth=2, label='7日移动平均')
    plt.axvline(x=train_df['date'].iloc[-1], color='k', linestyle='--', label='训练/测试分界')
    plt.xlabel('日期')
    plt.ylabel('每日新增确诊')
    plt.title('日本COVID-19每日新增确诊时间序列')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图1_数据概览.png', dpi=300)
    plt.close()

def plot_seir_fit():
    """图2：SEIR拟合结果"""
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
    seir_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv', parse_dates=['date'])

    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv', parse_dates=['date'])
    train_size = len(train_df)

    plt.figure(figsize=(14, 5))
    plt.plot(full_df['date'], full_df['new_confirmed_smooth'], 'b-', label='真实数据(平滑)')
    plt.plot(seir_results['date'], seir_results['predicted_new'], 'r--', label='SEIR拟合')
    plt.axvline(x=full_df['date'].iloc[train_size], color='k', linestyle=':', label='训练/测试分界')
    plt.xlabel('日期')
    plt.ylabel('每日新增确诊')
    plt.title('SEIR模型拟合效果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图2_SEIR拟合结果.png', dpi=300)
    plt.close()

def plot_seiqr_fit():
    """图3：SEIQR拟合 + 人群变化"""
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
    seiqr_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seiqr_results.csv', parse_dates=['date'])

    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv', parse_dates=['date'])
    train_size = len(train_df)

    # 子图1：拟合对比
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(full_df['date'], full_df['new_confirmed_smooth'], 'b-', label='真实数据')
    axes[0].plot(seiqr_results['date'], seiqr_results['predicted_new'], 'r--', label='SEIQR拟合')
    axes[0].axvline(x=full_df['date'].iloc[train_size], color='k', linestyle=':', label='训练/测试分界')
    axes[0].set_xlabel('日期')
    axes[0].set_ylabel('每日新增确诊')
    axes[0].set_title('SEIQR模型拟合效果')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：人群变化
    axes[1].plot(seiqr_results['date'], seiqr_results['S'] / 1e6, label='易感者 S (百万)', alpha=0.8)
    axes[1].plot(seiqr_results['date'], seiqr_results['E'] / 1e3, label='潜伏期 E (千)', alpha=0.8)
    axes[1].plot(seiqr_results['date'], seiqr_results['I'] / 1e3, label='感染者 I (千)', alpha=0.8)
    axes[1].plot(seiqr_results['date'], seiqr_results['Q'] / 1e3, label='隔离者 Q (千)', alpha=0.8)
    axes[1].plot(seiqr_results['date'], seiqr_results['R'] / 1e6, label='移除者 R (百万)', alpha=0.8)
    axes[1].set_xlabel('日期')
    axes[1].set_title('SEIQR人群变化趋势')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图3_SEIQR拟合结果.png', dpi=300)
    plt.close()

def plot_policy_simulation():
    """图4：政策模拟"""
    policy_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/policy_simulation.csv')

    plt.figure(figsize=(14, 5))
    plt.plot(policy_df['无隔离'], label='无隔离措施', linewidth=2)
    plt.plot(policy_df['轻度隔离'], label='轻度隔离 (δ=0.1)', linewidth=2)
    plt.plot(policy_df['严格隔离'], label='严格隔离 (δ=0.3)', linewidth=2)
    plt.xlabel('时间 (天)')
    plt.ylabel('感染人数 I(t)')
    plt.title('不同隔离政策对疫情发展的影响模拟')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图4_政策模拟.png', dpi=300)
    plt.close()

def plot_lstm_results():
    """图5：LSTM预测效果"""
    lstm_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_results.csv', parse_dates=['date'])

    plt.figure(figsize=(14, 5))
    plt.plot(lstm_results['date'], lstm_results['actual'], 'b-', label='真实值')
    plt.plot(lstm_results['date'], lstm_results['predicted'], 'r--', label='LSTM预测')
    plt.xlabel('日期')
    plt.ylabel('每日新增确诊')
    plt.title('LSTM模型预测效果 (测试集)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图5_LSTM预测效果.png', dpi=300)
    plt.close()


def plot_forecast():
    """图6：未来7天和14天预测"""
    # 读取数据
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
    forecast_7d = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_7d.csv', parse_dates=['date'])
    forecast_14d = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_14d.csv', parse_dates=['date'])

    plt.figure(figsize=(14, 5))

    # 历史数据（最近60天）
    recent = full_df.tail(60)
    plt.plot(recent['date'], recent['new_confirmed_smooth'], 'b-', label='历史数据', linewidth=2)

    # 7天预测（使用 forecast_7d）
    plt.plot(forecast_7d['date'], forecast_7d['predicted'], 'ro-', label='7天预测', markersize=8, linewidth=2)

    # 14天预测（使用 forecast_14d）
    plt.plot(forecast_14d['date'], forecast_14d['predicted'], 'gs-', label='14天预测', markersize=6, linewidth=2)

    plt.axvline(x=full_df['date'].iloc[-1], color='k', linestyle='--', label='当前时刻', linewidth=2)
    plt.xlabel('日期')
    plt.ylabel('每日新增确诊')
    plt.title('LSTM未来7天和14天预测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图6_未来预测.png', dpi=300)
    plt.close()


def plot_model_comparison():
    models = ['SEIR', 'SEIQR', 'LSTM']
    rmse = [117549.31, 117549.31, 15624.30]  # ← 更新
    mae = [98427.79, 98427.79, 9220.42]      # ← 更新
    r2 = [-2.3459, -2.3459, 0.9328]          # ← 更新

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # RMSE
    axes[0].bar(models, rmse, color=colors)
    axes[0].set_title('RMSE对比 (越低越好)')
    axes[0].set_ylabel('RMSE')
    for i, v in enumerate(rmse):
        axes[0].text(i, v + 2000, f'{v:.0f}', ha='center', fontsize=9)

    # MAE
    axes[1].bar(models, mae, color=colors)
    axes[1].set_title('MAE对比 (越低越好)')
    axes[1].set_ylabel('MAE')
    for i, v in enumerate(mae):
        axes[1].text(i, v + 2000, f'{v:.0f}', ha='center', fontsize=9)

    # R²
    axes[2].bar(models, r2, color=colors)
    axes[2].set_title('R²对比 (越高越好)')
    axes[2].set_ylabel('R²')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for i, v in enumerate(r2):
        if v < 0:
            axes[2].text(i, v - 0.1, f'{v:.2f}', ha='center', fontsize=9, color='red')
        else:
            axes[2].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图7_模型误差对比.png', dpi=300)
    plt.close()


def plot_sensitivity():
    """敏感性分析图：β±20%对峰值的影响"""
    df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/sensitivity_analysis.csv')

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df['场景'], df['峰值'], color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylabel('峰值感染人数')
    plt.title('感染率β对疫情峰值的影响（±20%敏感性分析）')
    for bar, v in zip(bars, df['峰值']):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 5000, f'{v:.0f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图_敏感性分析.png', dpi=300)
    plt.close()

def plot_lstm_loss():
    """LSTM训练损失曲线"""
    import pickle
    try:
        with open('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_history.pkl', 'rb') as f:
            history = pickle.load(f)
        plt.figure(figsize=(10, 4))
        plt.plot(history['loss'], label='训练损失')
        plt.plot(history['val_loss'], label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('LSTM训练损失收敛曲线')
        plt.tight_layout()
        plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图_LSTM训练损失.png', dpi=300)
        plt.close()
    except:
        print("未找到LSTM训练历史数据，跳过损失曲线图")

def main():
    print("生成可视化图表...")
    plot_data_overview()
    print("图1 数据概览")
    plot_seir_fit()
    print("图2 SEIR拟合")
    plot_seiqr_fit()
    print("图3 SEIQR拟合")
    plot_policy_simulation()
    print("图4 政策模拟")
    plot_lstm_results()
    print("图5 LSTM预测")
    plot_forecast()
    print("图6 未来预测")
    plot_model_comparison()
    print("图7 模型对比")
    plot_sensitivity()
    print("图8 敏感性分析")
    plot_lstm_loss()  # 可选
    print("所有图表生成完成！")
if __name__ == "__main__":
    main()