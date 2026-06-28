import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit     #曲线拟合（参数估计）
from sklearn.metrics import mean_squared_error, r2_score   #误差评估指标
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

'''
1. 首先读取数据，进行数据预处理（删除无效的数据），然后划分成53个训练集（1949年 - 2016年）和3个验证集（2021 - 2023年），
2. 训练模型：指数训练模型：通过最小二乘法，线性化后拟合得到参数  x0 = 61864.58, r = 0.013624
            改进指数训练模型：非线性最小二乘法直接拟合得到参数  x0=52371.76, r0=0.026033, a=0.00034579
            Logistic模型：非线性最小二乘法直接拟合得到参数  xm=158863.59, r=0.040096, x0=51932.01
3. 模型预测：对训练集预测（检查模型是否学会规律），然后对验证集预测（查看模型的泛化能力），最后对未来预测
4. 误差评估（拟合度检验R²、MSE、RMES、MAE、残差分析[隐含R²中]）：计算各个模型的误差
5. 将验证集的实际情与预测结果进行对比，预测未来结果（2024 - 2030年人口预测）
6. 绘图：左图为拟合效果图，右图为三个模型的未来预测
7. 找到最佳模型
'''

# 设置图片中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 读取数据
df = pd.read_excel("D:/datapackage/计算机建模/实验一/1949-2023人口数据-实验1.xlsx", skiprows=2)   #skiprows跳过前两行，因为表格中前两行是文字说明
df.columns = ["年份", "总人口", "男", "女"]   #给列重命名，方便后续使用

# 清洗总人口列：去除所有空白字符
df["总人口"] = df["总人口"].astype(str).str.strip()  # 去除首尾空格
df["总人口"] = df["总人口"].str.replace(r'\s+', '', regex=True)  # 去除所有空格
df["总人口"] = pd.to_numeric(df["总人口"], errors='coerce')

# 同样处理年份
df["年份"] = pd.to_numeric(df["年份"], errors='coerce')

df = df.dropna()
df = df[df["年份"] <= 2023]

# 确保数据类型为数值类型
df["年份"] = pd.to_numeric(df["年份"], errors='coerce')
df["总人口"] = pd.to_numeric(df["总人口"], errors='coerce')
df = df.dropna()

print(f"数据年份范围: {df['年份'].min()} - {df['年份'].max()}")
print(f"数据点数: {len(df)}")
print("\n数据样本:")
print(df.head(10))
print("\n最近数据:")
print(df.tail(10))

# 2. 划分训练集和验证集
# 注意：数据中只有2021、2022、2023年的数据
train = df[df["年份"] <= 2016].copy()   #1949 - 2016作为训练集，只有到2016的数据，然后就是2021 - 2023的数据
valid = df[df["年份"] >= 2021].copy()  # 2021 - 2023作为验证集

print(f"\n训练集: {len(train)} 个点 (年份: {train['年份'].min()}-{train['年份'].max()})")
print(f"验证集: {len(valid)} 个点 (年份: {valid['年份'].min()}-{valid['年份'].max()})")
print(f"验证集年份: {valid['年份'].values}")

# 确保是数值类型并转换为 numpy 数组，将年份转换为从0开始的时间，即1949 -> 0；.value.astype(float)确保数据是数值类型，避免类型错误
t_train = (train["年份"] - 1949).values.astype(float)
x_train = train["总人口"].values.astype(float)
t_valid = (valid["年份"] - 1949).values.astype(float)
x_valid = valid["总人口"].values.astype(float)

print(f"\nt_train 形状: {t_train.shape}")
print(f"x_train 形状: {x_train.shape}")
print(f"t_valid 形状: {t_valid.shape}")
print(f"x_valid 形状: {x_valid.shape}")


# 3. 指数增长模型 t:时间，x0:t = 0时候的人口数量，初始时只是使用实际生活中的1949年的人口，来帮助算法收敛，但是实际上不同的模型拟合出来的t=0时刻即1949年的人口是不一样的  r:自然增长率
def exp_model(t, x0, r):
    return x0 * np.exp(r * t)


# 线性化拟合    参数估计，找到最优的x0和r
log_x = np.log(x_train)   #对公式取对数，变成直线方程
coef = np.polyfit(t_train, log_x, 1)   #最小二乘法拟合直线
r_exp = coef[0]
x0_exp = np.exp(coef[1])

print(f"\n指数模型参数: x0 = {x0_exp:.2f}, r = {r_exp:.6f}")

# 预测
x_pred_exp_train = exp_model(t_train, x0_exp, r_exp)
x_pred_exp_valid = exp_model(t_valid, x0_exp, r_exp)


# 4. 改进指数增长模型, 允许增长率随时间变化不再是常数了   r0:初始增长率   a:增长率下降的速度
def improved_exp_model(t, x0, r0, a):
    return x0 * np.exp(r0 * t - 0.5 * a * t ** 2)


try:
    #参数估计     curve_fit:非线性最小二乘法，直接拟合曲线，   p0为初始猜测值，可以帮助算法更快找到最优解
    initial_guess = [x_train[0], r_exp, 0.0001]
    popt, _ = curve_fit(improved_exp_model, t_train, x_train, p0=initial_guess, maxfev=5000)

    x0_imp, r0_imp, a_imp = popt
    x_pred_imp_train = improved_exp_model(t_train, *popt)
    x_pred_imp_valid = improved_exp_model(t_valid, *popt)
    print(f"改进指数模型参数: x0={x0_imp:.2f}, r0={r0_imp:.6f}, a={a_imp:.8f}")
except Exception as e:
    print(f"改进指数模型拟合失败: {e}")
    x0_imp, r0_imp, a_imp = None, None, None
    x_pred_imp_valid = None


# 5. Logistic 模型， 考虑到资源限制，人口有上限 呈S形曲线：开始快速增长然后减速最后趋于饱和   xm:最大人口量（环境承载力）
def logistic_model(t, xm, r, x0):
    return xm / (1 + (xm / x0 - 1) * np.exp(-r * t))


try:
    initial_guess_log = [150000, 0.02, x_train[0]]
    bounds_log = ([100000, 0.001, 50000], [200000, 0.05, 60000])
    #参数估计，bounds是限制参数范围，比如说人口容量不可能低于10万
    popt_log, _ = curve_fit(logistic_model, t_train, x_train, p0=initial_guess_log, bounds=bounds_log, maxfev=5000)
    xm, r_log, x0_log = popt_log
    x_pred_log_train = logistic_model(t_train, *popt_log)
    x_pred_log_valid = logistic_model(t_valid, *popt_log)
    print(f"Logistic模型参数: xm={xm:.2f}, r={r_log:.6f}, x0={x0_log:.2f}")
except Exception as e:
    print(f"Logistic模型拟合失败: {e}")
    xm, r_log, x0_log = None, None, None
    x_pred_log_valid = None

# ========== 在验证集评估之前，添加训练集评估 ==========
print("\n" + "=" * 60)
print("训练集拟合度检验 (1949-2016年):")
print("=" * 60)

# 指数模型训练集评估
mse_exp_train = mean_squared_error(x_train, x_pred_exp_train)
rmse_exp_train = np.sqrt(mse_exp_train)
r2_exp_train = r2_score(x_train, x_pred_exp_train)
mae_exp_train = mean_absolute_error(x_train, x_pred_exp_train)

print(f"\n指数模型:")
print(f"  R² = {r2_exp_train:.4f}")
print(f"  RMSE = {rmse_exp_train:.2f} 万人")
print(f"  MAE = {mae_exp_train:.2f} 万人")
print(f"  MSE = {mse_exp_train:.2f}")

# 改进指数模型训练集评估
if x_pred_imp_train is not None:
    mse_imp_train = mean_squared_error(x_train, x_pred_imp_train)
    rmse_imp_train = np.sqrt(mse_imp_train)
    r2_imp_train = r2_score(x_train, x_pred_imp_train)
    mae_imp_train = mean_absolute_error(x_train, x_pred_imp_train)

    print(f"\n改进指数模型:")
    print(f"  R² = {r2_imp_train:.4f}")
    print(f"  RMSE = {rmse_imp_train:.2f} 万人")
    print(f"  MAE = {mae_imp_train:.2f} 万人")
    print(f"  MSE = {mse_imp_train:.2f}")

# Logistic模型训练集评估
if x_pred_log_train is not None:
    mse_log_train = mean_squared_error(x_train, x_pred_log_train)
    rmse_log_train = np.sqrt(mse_log_train)
    r2_log_train = r2_score(x_train, x_pred_log_train)
    mae_log_train = mean_absolute_error(x_train, x_pred_log_train)

    print(f"\nLogistic模型:")
    print(f"  R² = {r2_log_train:.4f}")
    print(f"  RMSE = {rmse_log_train:.2f} 万人")
    print(f"  MAE = {mae_log_train:.2f} 万人")
    print(f"  MSE = {mse_log_train:.2f}")

# 6. 误差评估
def evaluate(y_true, y_pred, model_name=""):
    if len(y_true) != len(y_pred):
        print(f"警告: {model_name} 长度不匹配 - 真实值:{len(y_true)}, 预测值:{len(y_pred)}")
        return None, None, None, None

    mse = mean_squared_error(y_true, y_pred)  #均方误差：误差平方的平均值，越大说明模型越差
    rmse = np.sqrt(mse)  #均方根误差  :MSE的平方根，更直观
    r2 = r2_score(y_true, y_pred)  #R² 决定系数 ：1表示完美拟合，0表示没比平均数好
    mae = np.mean(np.abs(y_true - y_pred))   #平均绝对误差  ： 平均预测误差
    return mse, rmse, r2, mae


print("\n" + "=" * 60)
print("验证集误差评估 (2021-2023年):")
print("=" * 60)

print(f"\n验证集实际数据:")
for year, pop in zip(valid["年份"].values, x_valid):
    print(f"  {year}年: {pop:.0f}万人")
#指数模型
print(f"\n指数模型预测:")
for year, pred in zip(valid["年份"].values, x_pred_exp_valid):
    print(f"  {year}年: {pred:.0f}万人")
    error_percent = abs(pred - x_valid[list(valid["年份"].values).index(year)]) / x_valid[
        list(valid["年份"].values).index(year)] * 100
    print(f"    误差: {error_percent:.2f}%")

# 指数模型的误差评估
results = evaluate(x_valid, x_pred_exp_valid, "指数模型")
if results[0] is not None:
    mse_exp, rmse_exp, r2_exp, mae_exp = results
    print(f"\n指数模型:")
    print(f"  MSE: {mse_exp:.2f}, RMSE: {rmse_exp:.2f}, MAE: {mae_exp:.2f}, R²: {r2_exp:.4f}")

# 改进指数模型
if x_pred_imp_valid is not None:
    print(f"\n改进指数模型预测:")
    for year, pred in zip(valid["年份"].values, x_pred_imp_valid):
        print(f"  {year}年: {pred:.0f}万人")
        error_percent = abs(pred - x_valid[list(valid["年份"].values).index(year)]) / x_valid[
            list(valid["年份"].values).index(year)] * 100
        print(f"    误差: {error_percent:.2f}%")
#改进指数模型的误差评估
    results = evaluate(x_valid, x_pred_imp_valid, "改进指数模型")
    if results[0] is not None:
        mse_imp, rmse_imp, r2_imp, mae_imp = results
        print(f"\n改进指数模型:")
        print(f"  MSE: {mse_imp:.2f}, RMSE: {rmse_imp:.2f}, MAE: {mae_imp:.2f}, R²: {r2_imp:.4f}")

# Logistic模型
if x_pred_log_valid is not None:
    print(f"\nLogistic模型预测:")
    for year, pred in zip(valid["年份"].values, x_pred_log_valid):
        print(f"  {year}年: {pred:.0f}万人")
        error_percent = abs(pred - x_valid[list(valid["年份"].values).index(year)]) / x_valid[
            list(valid["年份"].values).index(year)] * 100
        print(f"    误差: {error_percent:.2f}%")
#Logistic模型的误差评估
    results = evaluate(x_valid, x_pred_log_valid, "Logistic模型")
    if results[0] is not None:
        mse_log, rmse_log, r2_log, mae_log = results
        print(f"\nLogistic模型:")
        print(f"  MSE: {mse_log:.2f}, RMSE: {rmse_log:.2f}, MAE: {mae_log:.2f}, R²: {r2_log:.4f}")

# 7. 预测未来
future_years = np.arange(2024, 2031) - 1949    #2024 - 2030年的时间值
pred_exp_future = exp_model(future_years, x0_exp, r_exp)   #指数增长模型

print("\n" + "=" * 60)
print("2024-2030年人口预测（万人）:")
print("=" * 60)
print(f"{'年份':<8} {'指数模型':<12} {'改进指数模型':<12} {'Logistic模型':<12}")
print("-" * 50)

for i, year in enumerate(range(2024, 2031)):
    exp_pred = pred_exp_future[i]

    imp_pred = 0
    if x0_imp is not None:
        imp_pred = improved_exp_model(future_years[i], x0_imp, r0_imp, a_imp)  #改进指数增长模型

    log_pred = 0
    if xm is not None:
        log_pred = logistic_model(future_years[i], xm, r_log, x0_log)  #logistic模型

    print(f"{year:<8} {exp_pred:<12.0f} {imp_pred:<12.0f} {log_pred:<12.0f}")

# 8. 绘图 1行2列总大小15*6英寸
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 子图1：拟合效果
ax1 = axes[0]

# 绘制实际数据
ax1.plot(train["年份"], x_train, 'o', label='训练数据 (1949-2016)', markersize=4, color='blue')
ax1.plot(valid["年份"], x_valid, 's', label='验证数据 (2021-2023)', markersize=6, color='green')

# 指数模型
ax1.plot(train["年份"], x_pred_exp_train, '-', label='指数模型(训练)', linewidth=1.5, color='red')
ax1.plot(valid["年份"], x_pred_exp_valid, '--', label='指数模型(验证)', linewidth=1.5, color='red', alpha=0.7)

# 添加改进指数模型的训练曲线
if x_pred_imp_valid is not None:
    ax1.plot(train["年份"], x_pred_imp_train, '-', label='改进指数模型(训练)', linewidth=1.5, color='orange')
    ax1.plot(valid["年份"], x_pred_imp_valid, '--', label='改进指数模型(验证)', linewidth=1.5, color='orange', alpha=0.7)

# Logistic模型的训练曲线
if x_pred_log_valid is not None:
    ax1.plot(train["年份"], x_pred_log_train, '-', label='Logistic模型(训练)', linewidth=1.5, color='purple')
    ax1.plot(valid["年份"], x_pred_log_valid, '--', label='Logistic模型(验证)', linewidth=1.5, color='purple', alpha=0.7)

ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('总人口（万人）', fontsize=12)
ax1.set_title('人口增长模型拟合效果', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2：预测效果
ax2 = axes[1]
future_years_plot = np.arange(2024, 2031)

ax2.plot(train["年份"], x_train, 'o', label='历史数据 (1949-2016)', markersize=3, color='blue', alpha=0.5)
ax2.plot(valid["年份"], x_valid, 's', label='验证数据 (2021-2023)', markersize=5, color='green')

# 绘制预测
ax2.plot(future_years_plot, pred_exp_future, '--', label='指数模型预测', linewidth=2, color='red')

if x0_imp is not None:
    imp_future = [improved_exp_model(y - 1949, x0_imp, r0_imp, a_imp) for y in future_years_plot]
    ax2.plot(future_years_plot, imp_future, '--', label='改进指数模型预测', linewidth=2, color='orange')

if xm is not None:
    log_future = [logistic_model(y - 1949, xm, r_log, x0_log) for y in future_years_plot]
    ax2.plot(future_years_plot, log_future, '--', label='Logistic模型预测', linewidth=2, color='purple')

# 添加2023年的实际值作为参考
if len(x_valid) > 0:
    ax2.plot(valid["年份"].values[-1], x_valid[-1], 'ro', label='2023实际值', markersize=8)

ax2.set_xlabel('年份', fontsize=12)
ax2.set_ylabel('总人口（万人）', fontsize=12)
ax2.set_title('2024-2030年人口预测', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. 输出详细总结
print("\n" + "=" * 60)
print("模型对比总结:")
print("=" * 60)

print("\n1. 指数模型: 简单但长期预测可能不合理（持续增长）")
print(f"   - 预测2024年人口: {pred_exp_future[0]:.0f}万人")
print(f"   - 预测2030年人口: {pred_exp_future[-1]:.0f}万人")
print(f"   - 验证集平均误差: {mae_exp:.0f}万人" if 'mae_exp' in locals() else "")

if x0_imp is not None:
    print(f"\n2. 改进指数模型: 考虑增长率变化")
    print(f"   - 预测2024年人口: {improved_exp_model(future_years[0], x0_imp, r0_imp, a_imp):.0f}万人")
    print(f"   - 预测2030年人口: {improved_exp_model(future_years[-1], x0_imp, r0_imp, a_imp):.0f}万人")
    if 'mae_imp' in locals():
        print(f"   - 验证集平均误差: {mae_imp:.0f}万人")

if xm is not None:
    print(f"\n3. Logistic模型: 考虑人口容量")
    print(f"   - 预测2024年人口: {logistic_model(future_years[0], xm, r_log, x0_log):.0f}万人")
    print(f"   - 预测2030年人口: {logistic_model(future_years[-1], xm, r_log, x0_log):.0f}万人")
    print(f"   - 最大人口容量: {xm:.0f}万人")
    if 'mae_log' in locals():
        print(f"   - 验证集平均误差: {mae_log:.0f}万人")

# 找出最佳模型
errors = {}
if x_pred_exp_valid is not None and len(x_valid) == len(x_pred_exp_valid):
    errors['指数模型'] = mean_squared_error(x_valid, x_pred_exp_valid)

if x_pred_imp_valid is not None and len(x_valid) == len(x_pred_imp_valid):
    errors['改进指数模型'] = mean_squared_error(x_valid, x_pred_imp_valid)

if x_pred_log_valid is not None and len(x_valid) == len(x_pred_log_valid):
    errors['Logistic模型'] = mean_squared_error(x_valid, x_pred_log_valid)

if errors:
    best_model = min(errors, key=errors.get)
    print(f"\n在验证集上表现最好的模型是: {best_model}")
    print(f"其MSE为: {errors[best_model]:.2f}")