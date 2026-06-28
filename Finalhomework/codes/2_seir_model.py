# #定义SEIR和SEIQR微分方程 → 用训练集拟合参数 → 运行仿真 → 保存预测结果。
# """
# SEIR/SEIQR动力学模型模块
# 包含微分方程定义、参数拟合、模型求解
# """
# import numpy as np
# import pandas as pd
# from scipy.integrate import odeint
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# import os
#
#
# def seir_model(y, t, beta, sigma, gamma, N):
#     """SEIR模型微分方程"""
#     S, E, I, R = y
#     dS = -beta * S * I / N
#     dE = beta * S * I / N - sigma * E
#     dI = sigma * E - gamma * I
#     dR = gamma * I
#     return [dS, dE, dI, dR]
#
#
# def seiqr_model(y, t, beta, sigma, gamma, delta, gamma_q, theta, N):
#     """SEIQR模型微分方程（带隔离仓室）"""
#     S, E, I, Q, R = y
#     dS = -beta * S * (I + theta * E) / N
#     dE = beta * S * (I + theta * E) / N - sigma * E
#     dI = sigma * E - (delta + gamma) * I
#     dQ = delta * I - gamma_q * Q
#     dR = gamma * I + gamma_q * Q
#     return [dS, dE, dI, dQ, dR]
#
#
# def fit_seir_parameters(train_df, population, days_for_fit=60):
#     """用训练集前days_for_fit天拟合SEIR参数"""
#     fit_data = train_df.iloc[:days_for_fit]
#     # fit_data = train_df  #消切片，用全部训练数据拟合，轻微改善仿真曲线贴合度
#     # 初始化状态
#     first_case_idx = fit_data[fit_data['new_confirmed_smooth'] > 0].index[0]
#     I0 = fit_data.loc[first_case_idx, 'confirmed']
#     E0 = I0 * 1.5
#     R0 = fit_data.loc[first_case_idx, 'recovered']
#     S0 = population - I0 - E0 - R0
#
#     actual_cases = fit_data['new_confirmed_smooth'].values
#
#     def predict_cases(params):
#         beta, sigma, gamma = params
#         if not (0 < beta < 1 and 0 < sigma < 1 and 0 < gamma < 1):
#             return np.ones(len(actual_cases)) * 1e10
#
#         t = np.arange(len(actual_cases))
#         y0 = [S0, E0, I0, R0]
#         sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma, population))
#         I_pred = sol[:, 2]
#         # 修改为差分计算每日新增
#         daily_new = np.diff(I_pred, prepend=I_pred[0])
#         return daily_new
#
#     def loss(params):
#         pred = predict_cases(params)
#         # 修改：直接比较实际数值，而不是归一化后比较
#         # 只取前60天中疫情爆发后的部分
#         return np.mean((pred - actual_cases) ** 2)
#
#     # 使用更精确的优化方法
#     result = minimize(loss, [0.3, 0.2, 0.1],
#                       bounds=[(0.01, 0.9), (0.01, 0.5), (0.01, 0.5)],
#                       method='L-BFGS-B',
#                       options={'maxiter': 1000})
#
#     beta, sigma, gamma = result.x
#     return beta, sigma, gamma, S0, E0, I0, R0
#
# def run_seir_simulation(train_df, test_df, population, beta, sigma, gamma, S0, E0, I0, R0):
#     """运行SEIR全时段仿真"""
#     total_days = len(train_df) + len(test_df)
#     t = np.arange(total_days)
#     y0 = [S0, E0, I0, R0]
#
#     sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma, population))
#     S, E, I, R = sol.T
#
#     # 计算每日新增（从I的变化率）
#     daily_new = np.diff(I, prepend=I[0])
#     daily_new = np.maximum(daily_new, 0)  # 确保非负
#
#     # 构建结果DataFrame
#     full_df = pd.concat([train_df, test_df], ignore_index=True)
#     results = pd.DataFrame({
#         'date': full_df['date'],
#         'S': S,
#         'E': E,
#         'I': I,
#         'R': R,
#         'predicted_new': daily_new
#     })
#     return results
#
# def run_seiqr_simulation(train_df, test_df, population, beta, sigma, gamma, delta, gamma_q, theta, S0, E0, I0, R0):
#     """运行SEIQR全时段仿真"""
#     total_days = len(train_df) + len(test_df)
#     t = np.arange(total_days)
#     y0 = [S0, E0, I0, 0, R0]  # Q初始为0
#     sol = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, delta, gamma_q, theta, population))
#     S, E, I, Q, R = sol.T
#
#     daily_new = np.diff(I, prepend=I[0])
#     daily_new = np.maximum(daily_new, 0)  # 确保非负
#
#     full_df = pd.concat([train_df, test_df], ignore_index=True)
#     results = pd.DataFrame({
#         'date': full_df['date'],
#         'S': S,
#         'E': E,
#         'I': I,
#         'Q': Q,
#         'R': R,
#         'predicted_new': daily_new
#     })
#     return results
#
# def policy_simulation(population, beta, sigma, gamma, S0, E0, I0, R0, days=200):
#     """
#     政策模拟：对比不同隔离率下的疫情发展
#     """
#     t = np.arange(days)
#     # 场景1: 无隔离 (delta=0)
#     y0 = [S0, E0, I0, 0, R0]
#     sol1 = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, 0, 0.1, 0.5, population))
#     I1 = sol1[:, 2]
#     # 场景2: 轻度隔离 (delta=0.1)
#     sol2 = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, 0.1, 0.1, 0.5, population))
#     I2 = sol2[:, 2]
#     # 场景3: 严格隔离 (delta=0.3)
#     sol3 = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, 0.3, 0.1, 0.5, population))
#     I3 = sol3[:, 2]
#     return I1, I2, I3
#
# def main():
#     print("=" * 50)
#     print("SEIR/SEIQR模型仿真...")
#     # 加载预处理数据
#     train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv')
#     test_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_test.csv')
#
#     # 参数设置
#     POPULATION = 126000000  # 日本人口
#
#     # 拟合SEIR参数
#     beta, sigma, gamma, S0, E0, I0, R0 = fit_seir_parameters(train_df, POPULATION)
#     print(f"拟合参数: beta={beta:.4f}, sigma={sigma:.4f}, gamma={gamma:.4f}")
#     print(f"基本再生数 R0 = {beta / gamma:.2f}")
#
#     # 运行SEIR
#     results_seir = run_seir_simulation(train_df, test_df, POPULATION, beta, sigma, gamma, S0, E0, I0, R0)
#     results_seir.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv', index=False)
#     print("SEIR仿真完成")
#
#     # 运行SEIQR (使用相同的beta, sigma, gamma，增加隔离参数)
#     delta = 0.15
#     gamma_q = 0.1
#     theta = 0.5
#     results_seiqr = run_seiqr_simulation(train_df, test_df, POPULATION, beta, sigma, gamma, delta, gamma_q, theta, S0, E0, I0, R0)
#     results_seiqr.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seiqr_results.csv', index=False)
#     print("SEIQR仿真完成")
#
#     # 政策模拟
#     I1, I2, I3 = policy_simulation(POPULATION, beta, sigma, gamma, S0, E0, I0, R0)
#     policy_df = pd.DataFrame({
#         '无隔离': I1,
#         '轻度隔离': I2,
#         '严格隔离': I3
#     })
#     policy_df.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/policy_simulation.csv', index=False)
#     print("政策模拟完成")
#     print("=" * 50)
#
#     # ==========================================
#     # 计算SEIR和SEIQR的误差指标
#     # ==========================================
#     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#
#     # 加载完整数据
#     full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv')
#     train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv')
#     train_size = len(train_df)
#
#     # SEIR误差（在测试集上）
#     seir_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv')
#     test_actual = full_df['new_confirmed_smooth'].iloc[train_size:].values
#     test_pred_seir = seir_results['predicted_new'].iloc[train_size:].values
#
#     seir_rmse = np.sqrt(mean_squared_error(test_actual, test_pred_seir))
#     seir_mae = mean_absolute_error(test_actual, test_pred_seir)
#     seir_r2 = r2_score(test_actual, test_pred_seir)
#
#     # SEIQR误差（在测试集上）
#     seiqr_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seiqr_results.csv')
#     test_pred_seiqr = seiqr_results['predicted_new'].iloc[train_size:].values
#
#     seiqr_rmse = np.sqrt(mean_squared_error(test_actual, test_pred_seiqr))
#     seiqr_mae = mean_absolute_error(test_actual, test_pred_seiqr)
#     seiqr_r2 = r2_score(test_actual, test_pred_seiqr)
#
#     print("\n" + "=" * 50)
#     print("模型误差对比（测试集）")
#     print("=" * 50)
#     print(f"SEIR  - RMSE: {seir_rmse:.2f}, MAE: {seir_mae:.2f}, R²: {seir_r2:.4f}")
#     print(f"SEIQR - RMSE: {seiqr_rmse:.2f}, MAE: {seiqr_mae:.2f}, R²: {seiqr_r2:.4f}")
#     print("=" * 50)
#
# if __name__ == "__main__":
#     main()


"""
SEIR/SEIQR动力学模型模块
包含微分方程定义、参数拟合、模型求解
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def seir_model(y, t, beta, sigma, gamma, N):
    """SEIR模型微分方程"""
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


def seiqr_model(y, t, beta, sigma, gamma, delta, gamma_q, theta, N):
    """SEIQR模型微分方程（带隔离仓室）"""
    S, E, I, Q, R = y
    dS = -beta * S * (I + theta * E) / N
    dE = beta * S * (I + theta * E) / N - sigma * E
    dI = sigma * E - (delta + gamma) * I
    dQ = delta * I - gamma_q * Q
    dR = gamma * I + gamma_q * Q
    return [dS, dE, dI, dQ, dR]


def fit_seir_parameters(train_df, population, days_for_fit=60):
    """用训练集前days_for_fit天拟合SEIR参数"""
    fit_data = train_df.iloc[:days_for_fit]
    # 初始化状态
    first_case_idx = fit_data[fit_data['new_confirmed_smooth'] > 0].index[0]
    I0 = fit_data.loc[first_case_idx, 'confirmed']
    E0 = I0 * 1.5
    R0 = fit_data.loc[first_case_idx, 'recovered']
    S0 = population - I0 - E0 - R0

    actual_cases = fit_data['new_confirmed_smooth'].values

    def predict_cases(params):
        beta, sigma, gamma = params
        if not (0 < beta < 1 and 0 < sigma < 1 and 0 < gamma < 1):
            return np.ones(len(actual_cases)) * 1e10

        t = np.arange(len(actual_cases))
        y0 = [S0, E0, I0, R0]
        sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma, population))
        # 使用感染速率计算每日新增
        daily_new = beta * sol[:, 0] * sol[:, 2] / population
        daily_new = np.maximum(daily_new, 0)
        return daily_new

    def loss(params):
        pred = predict_cases(params)
        return np.mean((pred - actual_cases) ** 2)

    # result = minimize(loss, [0.3, 0.2, 0.1],
    #                   bounds=[(0.01, 0.9), (0.01, 0.5), (0.01, 0.5)],
    #                   method='L-BFGS-B',
    #                   options={'maxiter': 1000})
    result = minimize(loss, [0.3, 0.2, 0.1],
                      bounds=[(0.01, 0.9), (0.01, 0.9), (0.01, 0.9)],
                      method='L-BFGS-B',
                      options={'maxiter': 1000})

    beta, sigma, gamma = result.x
    return beta, sigma, gamma, S0, E0, I0, R0


def run_seir_simulation(train_df, test_df, population, beta, sigma, gamma, S0, E0, I0, R0):
    """运行SEIR全时段仿真"""
    total_days = len(train_df) + len(test_df)
    t = np.arange(total_days)
    y0 = [S0, E0, I0, R0]

    sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma, population))
    S, E, I, R = sol.T

    # 使用感染速率计算每日新增
    daily_new = beta * S * I / population
    daily_new = np.maximum(daily_new, 0)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    results = pd.DataFrame({
        'date': full_df['date'],
        'S': S,
        'E': E,
        'I': I,
        'R': R,
        'predicted_new': daily_new
    })
    return results


def run_seiqr_simulation(train_df, test_df, population, beta, sigma, gamma, delta, gamma_q, theta, S0, E0, I0, R0):
    """运行SEIQR全时段仿真"""
    total_days = len(train_df) + len(test_df)
    t = np.arange(total_days)
    y0 = [S0, E0, I0, 0, R0]
    sol = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, delta, gamma_q, theta, population))
    S, E, I, Q, R = sol.T

    # 使用感染速率计算每日新增
    daily_new = beta * S * (I + theta * E) / population
    daily_new = np.maximum(daily_new, 0)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    results = pd.DataFrame({
        'date': full_df['date'],
        'S': S,
        'E': E,
        'I': I,
        'Q': Q,
        'R': R,
        'predicted_new': daily_new
    })
    return results


def policy_simulation(population, beta, sigma, gamma, S0, E0, I0, R0, days=200):
    """政策模拟：对比不同隔离率下的疫情发展"""
    t = np.arange(days)
    y0 = [S0, E0, I0, 0, R0]
    sol1 = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, 0, 0.1, 0.5, population))
    I1 = sol1[:, 2]
    sol2 = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, 0.1, 0.1, 0.5, population))
    I2 = sol2[:, 2]
    sol3 = odeint(seiqr_model, y0, t, args=(beta, sigma, gamma, 0.3, 0.1, 0.5, population))
    I3 = sol3[:, 2]
    return I1, I2, I3


def sensitivity_analysis(population, beta, sigma, gamma, S0, E0, I0, R0, days=200):
    """敏感性分析：β±20%对峰值的影响"""
    betas = [beta * 0.8, beta, beta * 1.2]
    labels = ['β-20%', '原始β', 'β+20%']
    results = []

    for b, label in zip(betas, labels):
        t = np.arange(days)
        y0 = [S0, E0, I0, R0]
        sol = odeint(seir_model, y0, t, args=(b, sigma, gamma, population))
        I = sol[:, 2]
        peak = np.max(I)
        peak_time = np.argmax(I)
        results.append({'场景': label, '峰值': peak, '峰值时间(天)': peak_time})

    return pd.DataFrame(results)


def main():
    print("=" * 50)
    print("SEIR/SEIQR模型仿真...")

    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv')
    test_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_test.csv')

    POPULATION = 126000000

    beta, sigma, gamma, S0, E0, I0, R0 = fit_seir_parameters(train_df, POPULATION)
    print(f"拟合参数: beta={beta:.4f}, sigma={sigma:.4f}, gamma={gamma:.4f}")
    print(f"基本再生数 R0 = {beta / gamma:.2f}")

    # 运行SEIR
    results_seir = run_seir_simulation(train_df, test_df, POPULATION, beta, sigma, gamma, S0, E0, I0, R0)
    results_seir.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv', index=False)
    print("SEIR仿真完成")

    # 运行SEIQR
    delta = 0.15
    gamma_q = 0.1
    theta = 0.5
    results_seiqr = run_seiqr_simulation(train_df, test_df, POPULATION, beta, sigma, gamma, delta, gamma_q, theta, S0, E0, I0, R0)
    results_seiqr.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seiqr_results.csv', index=False)
    print("SEIQR仿真完成")

    # 政策模拟
    I1, I2, I3 = policy_simulation(POPULATION, beta, sigma, gamma, S0, E0, I0, R0)
    policy_df = pd.DataFrame({'无隔离': I1, '轻度隔离': I2, '严格隔离': I3})
    policy_df.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/policy_simulation.csv', index=False)
    print("政策模拟完成")

    # 敏感性分析
    sens_df = sensitivity_analysis(POPULATION, beta, sigma, gamma, S0, E0, I0, R0)
    sens_df.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/sensitivity_analysis.csv', index=False)
    print("敏感性分析完成")
    print(sens_df)

    # 计算SEIR和SEIQR误差
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv')
    train_size = len(train_df)

    seir_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv')
    test_actual = full_df['new_confirmed_smooth'].iloc[train_size:].values
    test_pred_seir = seir_results['predicted_new'].iloc[train_size:].values

    seir_rmse = np.sqrt(mean_squared_error(test_actual, test_pred_seir))
    seir_mae = mean_absolute_error(test_actual, test_pred_seir)
    seir_r2 = r2_score(test_actual, test_pred_seir)

    seiqr_results = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seiqr_results.csv')
    test_pred_seiqr = seiqr_results['predicted_new'].iloc[train_size:].values

    seiqr_rmse = np.sqrt(mean_squared_error(test_actual, test_pred_seiqr))
    seiqr_mae = mean_absolute_error(test_actual, test_pred_seiqr)
    seiqr_r2 = r2_score(test_actual, test_pred_seiqr)

    print("\n" + "=" * 50)
    print("模型误差对比（测试集）")
    print("=" * 50)
    print(f"SEIR  - RMSE: {seir_rmse:.2f}, MAE: {seir_mae:.2f}, R²: {seir_r2:.4f}")
    print(f"SEIQR - RMSE: {seiqr_rmse:.2f}, MAE: {seiqr_mae:.2f}, R²: {seiqr_r2:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()