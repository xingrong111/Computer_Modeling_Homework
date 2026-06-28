# """
# 可视化模块
# 生成所有论文用图
# """
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# # 设置中文字体（解决中文显示问题）
# import matplotlib
# matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# matplotlib.rcParams['axes.unicode_minus'] = False
#
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
#
# def plot_data_overview():
#     """图1：数据概览"""
#     df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
#     train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv', parse_dates=['date'])
#
#     plt.figure(figsize=(14, 5))
#     plt.plot(df['date'], df['new_confirmed'], alpha=0.4, label='原始每日新增')
#     plt.plot(df['date'], df['new_confirmed_smooth'], 'r-', linewidth=2, label='7日移动平均')
#     plt.axvline(x=train_df['date'].iloc[-1], color='k', linestyle='--', label='训练/测试分界')
#     plt.xlabel('日期')
#     plt.ylabel('每日新增确诊')
#     plt.title('日本COVID-19每日新增确诊时间序列')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图1_数据概览.png', dpi=300)
#     plt.close()
#
# def plot_seir_fit():
#     """图2：SEIR拟合结果"""
#     full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
#     seir_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv', parse_dates=['date'])
#
#     train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv', parse_dates=['date'])
#     train_size = len(train_df)
#
#     plt.figure(figsize=(14, 5))
#     plt.plot(full_df['date'], full_df['new_confirmed_smooth'], 'b-', label='真实数据(平滑)')
#     plt.plot(seir_results['date'], seir_results['predicted_new'], 'r--', label='SEIR拟合')
#     plt.axvline(x=full_df['date'].iloc[train_size], color='k', linestyle=':', label='训练/测试分界')
#     plt.xlabel('日期')
#     plt.ylabel('每日新增确诊')
#     plt.title('SEIR模型拟合效果')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图2_SEIR拟合结果.png', dpi=300)
#     plt.close()
#
# def plot_seiqr_fit():
#     """图3：SEIQR拟合 + 人群变化"""
#     full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
#     seiqr_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seiqr_results.csv', parse_dates=['date'])
#
#     train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv', parse_dates=['date'])
#     train_size = len(train_df)
#
#     # 子图1：拟合对比
#     fig, axes = plt.subplots(1, 2, figsize=(16, 5))
#
#     axes[0].plot(full_df['date'], full_df['new_confirmed_smooth'], 'b-', label='真实数据')
#     axes[0].plot(seiqr_results['date'], seiqr_results['predicted_new'], 'r--', label='SEIQR拟合')
#     axes[0].axvline(x=full_df['date'].iloc[train_size], color='k', linestyle=':', label='训练/测试分界')
#     axes[0].set_xlabel('日期')
#     axes[0].set_ylabel('每日新增确诊')
#     axes[0].set_title('SEIQR模型拟合效果')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
#
#     # 子图2：人群变化
#     axes[1].plot(seiqr_results['date'], seiqr_results['S'] / 1e6, label='易感者 S (百万)', alpha=0.8)
#     axes[1].plot(seiqr_results['date'], seiqr_results['E'] / 1e3, label='潜伏期 E (千)', alpha=0.8)
#     axes[1].plot(seiqr_results['date'], seiqr_results['I'] / 1e3, label='感染者 I (千)', alpha=0.8)
#     axes[1].plot(seiqr_results['date'], seiqr_results['Q'] / 1e3, label='隔离者 Q (千)', alpha=0.8)
#     axes[1].plot(seiqr_results['date'], seiqr_results['R'] / 1e6, label='移除者 R (百万)', alpha=0.8)
#     axes[1].set_xlabel('日期')
#     axes[1].set_title('SEIQR人群变化趋势')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图3_SEIQR拟合结果.png', dpi=300)
#     plt.close()
#
# def plot_policy_simulation():
#     """图4：政策模拟"""
#     policy_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/policy_simulation.csv')
#
#     plt.figure(figsize=(14, 5))
#     plt.plot(policy_df['无隔离'], label='无隔离措施', linewidth=2)
#     plt.plot(policy_df['轻度隔离'], label='轻度隔离 (δ=0.1)', linewidth=2)
#     plt.plot(policy_df['严格隔离'], label='严格隔离 (δ=0.3)', linewidth=2)
#     plt.xlabel('时间 (天)')
#     plt.ylabel('感染人数 I(t)')
#     plt.title('不同隔离政策对疫情发展的影响模拟')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图4_政策模拟.png', dpi=300)
#     plt.close()
#
# def plot_lstm_results():
#     """图5：LSTM预测效果"""
#     lstm_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_results.csv', parse_dates=['date'])
#
#     plt.figure(figsize=(14, 5))
#     plt.plot(lstm_results['date'], lstm_results['actual'], 'b-', label='真实值')
#     plt.plot(lstm_results['date'], lstm_results['predicted'], 'r--', label='LSTM预测')
#     plt.xlabel('日期')
#     plt.ylabel('每日新增确诊')
#     plt.title('LSTM模型预测效果 (测试集)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图5_LSTM预测效果.png', dpi=300)
#     plt.close()
#
#
# def plot_forecast():
#     """图6：未来7天和14天预测"""
#     # 读取数据
#     full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv', parse_dates=['date'])
#     forecast_7d = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_7d.csv', parse_dates=['date'])
#     forecast_14d = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_14d.csv', parse_dates=['date'])
#
#     plt.figure(figsize=(14, 5))
#
#     # 历史数据（最近60天）
#     recent = full_df.tail(60)
#     plt.plot(recent['date'], recent['new_confirmed_smooth'], 'b-', label='历史数据', linewidth=2)
#
#     # 7天预测（使用 forecast_7d）
#     plt.plot(forecast_7d['date'], forecast_7d['predicted'], 'ro-', label='7天预测', markersize=8, linewidth=2)
#
#     # 14天预测（使用 forecast_14d）
#     plt.plot(forecast_14d['date'], forecast_14d['predicted'], 'gs-', label='14天预测', markersize=6, linewidth=2)
#
#     plt.axvline(x=full_df['date'].iloc[-1], color='k', linestyle='--', label='当前时刻', linewidth=2)
#     plt.xlabel('日期')
#     plt.ylabel('每日新增确诊')
#     plt.title('LSTM未来7天和14天预测')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图6_未来预测.png', dpi=300)
#     plt.close()
#
#
# def plot_model_comparison():
#     models = ['SEIR', 'SEIQR', 'LSTM']
#     rmse = [117549.31, 117549.31, 15624.30]  # ← 更新
#     mae = [98427.79, 98427.79, 9220.42]      # ← 更新
#     r2 = [-2.3459, -2.3459, 0.9328]          # ← 更新
#
#     fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#     colors = ['#3498db', '#2ecc71', '#e74c3c']
#
#     # RMSE
#     axes[0].bar(models, rmse, color=colors)
#     axes[0].set_title('RMSE对比 (越低越好)')
#     axes[0].set_ylabel('RMSE')
#     for i, v in enumerate(rmse):
#         axes[0].text(i, v + 2000, f'{v:.0f}', ha='center', fontsize=9)
#
#     # MAE
#     axes[1].bar(models, mae, color=colors)
#     axes[1].set_title('MAE对比 (越低越好)')
#     axes[1].set_ylabel('MAE')
#     for i, v in enumerate(mae):
#         axes[1].text(i, v + 2000, f'{v:.0f}', ha='center', fontsize=9)
#
#     # R²
#     axes[2].bar(models, r2, color=colors)
#     axes[2].set_title('R²对比 (越高越好)')
#     axes[2].set_ylabel('R²')
#     axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
#     for i, v in enumerate(r2):
#         if v < 0:
#             axes[2].text(i, v - 0.1, f'{v:.2f}', ha='center', fontsize=9, color='red')
#         else:
#             axes[2].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9, color='green')
#
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图7_模型误差对比.png', dpi=300)
#     plt.close()
#
#
# def plot_sensitivity():
#     """敏感性分析图：β±20%对峰值的影响"""
#     df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/sensitivity_analysis.csv')
#
#     plt.figure(figsize=(10, 5))
#     bars = plt.bar(df['场景'], df['峰值'], color=['#3498db', '#2ecc71', '#e74c3c'])
#     plt.ylabel('峰值感染人数')
#     plt.title('感染率β对疫情峰值的影响（±20%敏感性分析）')
#     for bar, v in zip(bars, df['峰值']):
#         plt.text(bar.get_x() + bar.get_width() / 2, v + 5000, f'{v:.0f}', ha='center', fontsize=10)
#     plt.tight_layout()
#     plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图_敏感性分析.png', dpi=300)
#     plt.close()
#
# def plot_lstm_loss():
#     """LSTM训练损失曲线"""
#     import pickle
#     try:
#         with open('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_history.pkl', 'rb') as f:
#             history = pickle.load(f)
#         plt.figure(figsize=(10, 4))
#         plt.plot(history['loss'], label='训练损失')
#         plt.plot(history['val_loss'], label='验证损失')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.title('LSTM训练损失收敛曲线')
#         plt.tight_layout()
#         plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图_LSTM训练损失.png', dpi=300)
#         plt.close()
#     except:
#         print("未找到LSTM训练历史数据，跳过损失曲线图")
#
# def main():
#     print("生成可视化图表...")
#     plot_data_overview()
#     print("图1 数据概览")
#     plot_seir_fit()
#     print("图2 SEIR拟合")
#     plot_seiqr_fit()
#     print("图3 SEIQR拟合")
#     plot_policy_simulation()
#     print("图4 政策模拟")
#     plot_lstm_results()
#     print("图5 LSTM预测")
#     plot_forecast()
#     print("图6 未来预测")
#     plot_model_comparison()
#     print("图7 模型对比")
#     plot_sensitivity()
#     print("图8 敏感性分析")
#     plot_lstm_loss()  # 可选
#     print("所有图表生成完成！")
# if __name__ == "__main__":
#     main()


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

import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======================== 原有函数（保留） ========================

def plot_data_overview():
    """图1：数据概览"""
    df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv',
                     parse_dates=['date'])
    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv',
                           parse_dates=['date'])

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
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv',
                          parse_dates=['date'])
    seir_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv',
                               parse_dates=['date'])

    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv',
                           parse_dates=['date'])
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
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv',
                          parse_dates=['date'])
    seiqr_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seiqr_results.csv',
                                parse_dates=['date'])

    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv',
                           parse_dates=['date'])
    train_size = len(train_df)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(full_df['date'], full_df['new_confirmed_smooth'], 'b-', label='真实数据')
    axes[0].plot(seiqr_results['date'], seiqr_results['predicted_new'], 'r--', label='SEIQR拟合')
    axes[0].axvline(x=full_df['date'].iloc[train_size], color='k', linestyle=':', label='训练/测试分界')
    axes[0].set_xlabel('日期')
    axes[0].set_ylabel('每日新增确诊')
    axes[0].set_title('SEIQR模型拟合效果')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

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
    """图5：LSTM预测效果（单变量确诊）"""
    lstm_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_results.csv',
                               parse_dates=['date'])

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
    """图6：未来7天和14天预测（单变量确诊）"""
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv',
                          parse_dates=['date'])
    forecast_7d = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_7d.csv',
                              parse_dates=['date'])
    forecast_14d = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_14d.csv',
                               parse_dates=['date'])

    plt.figure(figsize=(14, 5))

    recent = full_df.tail(60)
    plt.plot(recent['date'], recent['new_confirmed_smooth'], 'b-', label='历史数据', linewidth=2)

    plt.plot(forecast_7d['date'], forecast_7d['predicted'], 'ro-', label='7天预测', markersize=8, linewidth=2)
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
    """图7：多模型误差对比"""
    models = ['SEIR', 'SEIQR', 'LSTM']
    rmse = [117549.31, 117549.31, 22385.41]  # ← 更新
    mae = [98427.79, 98427.79, 17121.98]  # ← 更新
    r2 = [-2.3459, -2.3459, 0.8621]  # ← 更新

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    axes[0].bar(models, rmse, color=colors)
    axes[0].set_title('RMSE对比 (越低越好)')
    axes[0].set_ylabel('RMSE')
    for i, v in enumerate(rmse):
        axes[0].text(i, v + 2000, f'{v:.0f}', ha='center', fontsize=9)

    axes[1].bar(models, mae, color=colors)
    axes[1].set_title('MAE对比 (越低越好)')
    axes[1].set_ylabel('MAE')
    for i, v in enumerate(mae):
        axes[1].text(i, v + 2000, f'{v:.0f}', ha='center', fontsize=9)

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
    """图8：敏感性分析"""
    df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/sensitivity_analysis.csv')

    plt.figure(figsize=(10, 5))
    bars = plt.bar(df['场景'], df['峰值'], color=['#3498db', '#2ecc71', '#e74c3c'])
    plt.ylabel('峰值感染人数')
    plt.title('感染率β对疫情峰值的影响（±20%敏感性分析）')
    for bar, v in zip(bars, df['峰值']):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 5000, f'{v:.0f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图8_敏感性分析.png', dpi=300)
    plt.close()


# ======================== 新增函数 ========================

def plot_region_comparison():
    """图9：四国新增确诊趋势对比（区域传播趋势分析）"""
    DATA_DIR = 'E:/计算机建模技术大作业内容-202606/COVID-19-数据/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/'
    confirmed = pd.read_csv(DATA_DIR + 'time_series_covid19_confirmed_global.csv')

    countries = {
        'Japan': '日本',
        'Korea, South': '韩国',
        'US': '美国',
        'Germany': '德国'
    }

    plt.figure(figsize=(14, 6))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for i, (eng_name, cn_name) in enumerate(countries.items()):
        country_df = confirmed[confirmed['Country/Region'] == eng_name]
        if country_df.shape[0] > 1:
            series = country_df.iloc[:, 4:].sum(axis=0)
        else:
            series = country_df.iloc[0, 4:]

        daily_new = series.diff().fillna(0).clip(lower=0)
        daily_smooth = daily_new.rolling(window=7, center=True).mean()
        dates = pd.to_datetime(series.index, errors='coerce')

        plt.plot(dates, daily_smooth, label=cn_name, color=colors[i], linewidth=2, alpha=0.85)

    plt.yscale('log')
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('每日新增确诊（对数刻度）', fontsize=12)
    plt.title('日本、韩国、美国、德国每日新增确诊趋势对比', fontsize=14)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.text(0.02, 0.98, '注：纵轴为对数刻度', transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图9_四国区域对比.png', dpi=300)
    plt.close()
    print("图9 四国区域对比已生成")


def plot_lstm_multi_results():
    """图10：LSTM对确诊+治愈+死亡的预测效果"""
    try:
        lstm_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_results_multi.csv',
                                   parse_dates=['date'])
    except FileNotFoundError:
        print("请先运行 3_lstm_model.py 生成多变量预测结果")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # 确诊
    axes[0].plot(lstm_results['date'], lstm_results['actual_confirmed'], 'b-', label='真实确诊', linewidth=2)
    axes[0].plot(lstm_results['date'], lstm_results['predicted_confirmed'], 'r--', label='预测确诊', linewidth=2)
    axes[0].set_ylabel('确诊人数（例/日）', fontsize=12)
    axes[0].set_title('LSTM确诊预测效果', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 治愈
    axes[1].plot(lstm_results['date'], lstm_results['actual_recovered'], 'g-', label='真实治愈', linewidth=2)
    axes[1].plot(lstm_results['date'], lstm_results['predicted_recovered'], 'orange', linestyle='--', label='预测治愈',
                 linewidth=2)
    axes[1].set_ylabel('治愈人数（例/日）', fontsize=12)
    axes[1].set_title('LSTM治愈预测效果', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 死亡
    axes[2].plot(lstm_results['date'], lstm_results['actual_deaths'], 'purple', label='真实死亡', linewidth=2)
    axes[2].plot(lstm_results['date'], lstm_results['predicted_deaths'], 'red', linestyle='--', label='预测死亡',
                 linewidth=2)
    axes[2].set_ylabel('死亡人数（例/日）', fontsize=12)
    axes[2].set_xlabel('日期', fontsize=12)
    axes[2].set_title('LSTM死亡预测效果', fontsize=13)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图10_治愈死亡预测.png', dpi=300)
    plt.close()
    print("图10 治愈死亡预测已生成")


def plot_forecast_multi():
    """图11：未来7天和14天确诊+治愈+死亡预测"""
    try:
        full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv',
                              parse_dates=['date'])
        forecast_7d = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_multi_7d.csv',
                                  parse_dates=['date'])
        forecast_14d = pd.read_csv(
            'D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_multi_14d.csv',
            parse_dates=['date'])
    except FileNotFoundError:
        print("请先运行 3_lstm_model.py 生成多变量预测结果")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    recent = full_df.tail(60)

    # 确诊
    axes[0].plot(recent['date'], recent['new_confirmed_smooth'], 'b-', label='历史确诊', linewidth=2)
    axes[0].plot(forecast_7d['date'], forecast_7d['confirmed'], 'ro-', label='7天预测', markersize=8, linewidth=2)
    axes[0].plot(forecast_14d['date'], forecast_14d['confirmed'], 'gs-', label='14天预测', markersize=6, linewidth=2)
    axes[0].axvline(x=full_df['date'].iloc[-1], color='k', linestyle='--', label='当前时刻')
    axes[0].set_ylabel('确诊人数', fontsize=12)
    axes[0].set_title('未来确诊预测', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 治愈
    axes[1].plot(recent['date'], recent['new_recovered_smooth'], 'g-', label='历史治愈', linewidth=2)
    axes[1].plot(forecast_7d['date'], forecast_7d['recovered'], 'ro-', label='7天预测', markersize=8, linewidth=2)
    axes[1].plot(forecast_14d['date'], forecast_14d['recovered'], 'gs-', label='14天预测', markersize=6, linewidth=2)
    axes[1].axvline(x=full_df['date'].iloc[-1], color='k', linestyle='--')
    axes[1].set_ylabel('治愈人数', fontsize=12)
    axes[1].set_title('未来治愈预测', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 死亡
    axes[2].plot(recent['date'], recent['new_deaths_smooth'], 'purple', label='历史死亡', linewidth=2)
    axes[2].plot(forecast_7d['date'], forecast_7d['deaths'], 'ro-', label='7天预测', markersize=8, linewidth=2)
    axes[2].plot(forecast_14d['date'], forecast_14d['deaths'], 'gs-', label='14天预测', markersize=6, linewidth=2)
    axes[2].axvline(x=full_df['date'].iloc[-1], color='k', linestyle='--')
    axes[2].set_ylabel('死亡人数', fontsize=12)
    axes[2].set_xlabel('日期', fontsize=12)
    axes[2].set_title('未来死亡预测', fontsize=13)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图11_未来三指标预测.png', dpi=300)
    plt.close()
    print("图11 未来三指标预测已生成")


def plot_model_comparison_multi():
    """图12：多模型误差对比（包含确诊/治愈/死亡）"""
    try:
        lstm_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_results_multi.csv')
        # 计算治愈和死亡的RMSE/MAE/R2
        y_actual_recovered = lstm_results['actual_recovered'].values
        y_pred_recovered = lstm_results['predicted_recovered'].values
        y_actual_deaths = lstm_results['actual_deaths'].values
        y_pred_deaths = lstm_results['predicted_deaths'].values

        rec_rmse = np.sqrt(mean_squared_error(y_actual_recovered, y_pred_recovered))
        rec_mae = mean_absolute_error(y_actual_recovered, y_pred_recovered)
        rec_r2 = r2_score(y_actual_recovered, y_pred_recovered)

        death_rmse = np.sqrt(mean_squared_error(y_actual_deaths, y_pred_deaths))
        death_mae = mean_absolute_error(y_actual_deaths, y_pred_deaths)
        death_r2 = r2_score(y_actual_deaths, y_pred_deaths)
    except:
        print("无法读取多变量预测结果，使用默认值")
        rec_rmse, rec_mae, rec_r2 = 0, 0, 0
        death_rmse, death_mae, death_r2 = 0, 0, 0

    models = ['SEIR', 'SEIQR', 'LSTM-确诊', 'LSTM-治愈', 'LSTM-死亡']
    rmse = [117549.31, 117549.31, 22385.41, rec_rmse, death_rmse]
    mae = [98427.79, 98427.79, 17121.98, rec_mae, death_mae]
    r2 = [-2.3459, -2.3459, 0.8621, rec_r2, death_r2]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for ax, data, title, ylabel in zip(
            axes,
            [rmse, mae, r2],
            ['RMSE对比 (越低越好)', 'MAE对比 (越低越好)', 'R²对比 (越高越好)'],
            ['RMSE', 'MAE', 'R²']
    ):
        bars = ax.bar(models, data, color=colors)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=12)
        for j, v in enumerate(data):
            if isinstance(v, float) and v < 1 and v > -10:
                ax.text(j, v + (0.02 if v >= 0 else -0.08), f'{v:.2f}', ha='center', fontsize=9)
            else:
                ax.text(j, v + (abs(v) * 0.02) + 100, f'{v:.0f}', ha='center', fontsize=9)
        if title == 'R²对比 (越高越好)':
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/图12_多变量模型对比.png', dpi=300)
    plt.close()
    print("图12 多变量模型对比已生成")


# ======================== Main 函数 ========================

def main():
    print("=" * 60)
    print("生成可视化图表...")
    print("=" * 60)

    plot_data_overview()
    print("✅ 图1 数据概览")
    plot_seir_fit()
    print("✅ 图2 SEIR拟合")
    plot_seiqr_fit()
    print("✅ 图3 SEIQR拟合")
    plot_policy_simulation()
    print("✅ 图4 政策模拟")
    plot_lstm_results()
    print("✅ 图5 LSTM预测")
    plot_forecast()
    print("✅ 图6 未来预测")
    plot_model_comparison()
    print("✅ 图7 模型对比")
    plot_sensitivity()
    print("✅ 图8 敏感性分析")

    # 新增图表
    plot_region_comparison()
    print("✅ 图9 四国区域对比")
    plot_lstm_multi_results()
    print("✅ 图10 治愈死亡预测")
    plot_forecast_multi()
    print("✅ 图11 未来三指标预测")
    plot_model_comparison_multi()
    print("✅ 图12 多变量模型对比")

    print("=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()