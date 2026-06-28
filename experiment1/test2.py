import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ========== 1. 中文显示设置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ========== 2. 数据预处理函数 ==========
def load_and_clean_data(filepath):
    """
    加载并清洗Excel数据

    步骤:
    1. 读取Excel，跳过说明行
    2. 重命名列
    3. 去除所有空白字符
    4. 转换数据类型
    5. 删除无效行
    6. 检查数据完整性
    """

    print("=" * 60)
    print("数据预处理开始")
    print("=" * 60)

    # 2.1 读取数据
    df = pd.read_excel(filepath, skiprows=2)
    df.columns = ["年份", "总人口", "男", "女"]
    print(f"✓ 原始数据行数: {len(df)}")

    # 2.2 去除所有空白字符（关键！）
    # 将每一列都转为字符串，去除首尾空格和中间特殊空格
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()  # 去除首尾空格
        df[col] = df[col].str.replace(r'\s+', '', regex=True)  # 去除所有空白字符
        df[col] = df[col].str.replace(r'\u00a0', '', regex=True)  # 去除不间断空格
        df[col] = df[col].str.replace(r'\t', '', regex=True)  # 去除制表符

    print(f"✓ 已去除所有空白字符")

    # 2.3 转换数据类型
    df["年份"] = pd.to_numeric(df["年份"], errors='coerce')
    df["总人口"] = pd.to_numeric(df["总人口"], errors='coerce')
    print(f"✓ 数据类型转换完成")

    # 2.4 检查无效数据
    print(f"\n无效数据统计:")
    print(f"  年份无效: {df['年份'].isna().sum()} 行")
    print(f"  总人口无效: {df['总人口'].isna().sum()} 行")

    # 2.5 删除无效行
    df = df.dropna(subset=["年份", "总人口"])
    print(f"✓ 删除无效行后: {len(df)} 行")

    # 2.6 确保年份为整数
    df["年份"] = df["年份"].astype(int)

    # 2.7 按年份排序
    df = df.sort_values("年份").reset_index(drop=True)

    # # 2.8 检查数据连续性
    # years = df["年份"].values
    # year_diff = np.diff(years)
    # missing_years = []
    # for i, diff in enumerate(year_diff):
    #     if diff > 1:
    #         for missing in range(years[i] + 1, years[i + 1]):
    #             missing_years.append(missing)

    # if missing_years:
    #     print(f"\n⚠ 发现缺失年份:")
    #     for year in missing_years:
    #         print(f"    {year}年数据缺失")
    # else:
    #     print(f"\n✓ 年份连续，无缺失")

    # 2.9 显示数据预览
    print(f"\n数据预览 (前5行):")
    print(df.head())
    print(f"\n数据预览 (后5行):")
    print(df.tail())

    print("\n" + "=" * 60)
    print("数据预处理完成")
    print("=" * 60)

    return df


# ========== 3. 灰色预测模型类 ==========
class GreyModel:
    """
    灰色预测模型 GM(1,1)
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.x0 = None
        self.x1 = None
        self.fitted_values = None

    def fit(self, x0):
        """拟合模型"""
        self.x0 = np.array(x0, dtype=float)
        n = len(self.x0)

        # 数据预检查
        if n < 4:
            raise ValueError(f"数据点太少（{n}个），灰色模型至少需要4个数据点")

        # 检查数据是否全为正数
        if np.any(self.x0 <= 0):
            print(f"⚠ 警告：数据中存在非正数")

        # 累加生成
        self.x1 = np.cumsum(self.x0)

        # 构造矩阵
        B = np.zeros((n - 1, 2))
        Y = np.zeros((n - 1, 1))

        for k in range(1, n):
            B[k - 1, 0] = -0.5 * (self.x1[k - 1] + self.x1[k])
            B[k - 1, 1] = 1
            Y[k - 1, 0] = self.x0[k]

        # 最小二乘估计
        BTB = np.dot(B.T, B)
        BTB_inv = np.linalg.inv(BTB)
        BTY = np.dot(B.T, Y)
        params = np.dot(BTB_inv, BTY)

        self.a = params[0, 0]
        self.b = params[1, 0]

        # 计算拟合值
        self.fitted_values = self._predict_internal(np.arange(1, n + 1))

        return self

    def _predict_internal(self, k_list):
        """内部预测函数"""
        pred = np.zeros(len(k_list))
        for i, k in enumerate(k_list):
            if k == 1:
                pred[i] = self.x0[0]
            else:
                pred[i] = (self.x0[0] - self.b / self.a) * (1 - np.exp(self.a)) * np.exp(-self.a * (k - 1))
        return pred

    def predict(self, n_steps):
        """预测未来n_steps步"""
        n = len(self.x0)
        k_list = n + np.arange(1, n_steps + 1)
        return self._predict_internal(k_list)

    def get_params(self):
        return {'a': self.a, 'b': self.b}


# ========== 4. 评估函数 ==========
def evaluate(y_true, y_pred, name=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} 评估:")
    print(f"   MSE: {mse:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   R²: {r2:.6f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


# ========== 5. 主程序 ==========
if __name__ == "__main__":

    # 5.1 加载并清洗数据
    df = load_and_clean_data("D:/datapackage/计算机建模/实验一/1949-2023人口数据-实验1.xlsx")

    # # 5.2 验证2022年数据是否存在
    # print(f"\n验证2022年数据:")
    # year_2022 = df[df["年份"] == 2022]
    # if len(year_2022) > 0:
    #     print(f"  ✓ 2022年数据存在: {year_2022['总人口'].iloc[0]} 万人")
    # else:
    #     print(f" 2022年数据缺失，需要手动补充")

    # 5.3 划分训练集和验证集
    train = df[df["年份"] <= 2016].copy()
    valid = df[df["年份"] >= 2021].copy()

    x_train = train["总人口"].values
    years_train = train["年份"].values
    x_valid = valid["总人口"].values
    years_valid = valid["年份"].values

    print(f"\n数据集划分:")
    print(f"  训练集: {len(x_train)} 个点 (1949-2016)")
    print(f"  验证集: {len(x_valid)} 个点 ({', '.join(map(str, years_valid))})")

    # 5.4 训练灰色模型
    print("\n" + "=" * 60)
    print("灰色预测模型训练")
    print("=" * 60)

    gm = GreyModel()
    gm.fit(x_train)

    params = gm.get_params()
    print(f"\n模型参数:")
    print(f"  发展系数 a = {params['a']:.6f}")
    print(f"  灰色作用量 b = {params['b']:.2f}")

    # 模型有效性检验
    development_coef = -params['a']
    if development_coef < 0.3:
        print(f"模型有效 (发展系数 = {development_coef:.4f} < 0.3)")
    elif development_coef < 0.5:
        print(f"模型勉强可用 (发展系数 = {development_coef:.4f})")
    else:
        print(f"模型无效 (发展系数 = {development_coef:.4f} > 0.5)")

    # 5.5 预测
    x_pred_train = gm._predict_internal(np.arange(1, len(x_train) + 1))
    x_pred_valid = gm.predict(len(x_valid))

    # 5.6 评估
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)

    train_metrics = evaluate(x_train, x_pred_train, "训练集")
    valid_metrics = evaluate(x_valid, x_pred_valid, "验证集")

    # 5.7 详细预测结果
    print(f"\n验证集详细预测:")
    for i, (year, actual, pred) in enumerate(zip(years_valid, x_valid, x_pred_valid)):
        error = actual - pred
        error_pct = abs(error) / actual * 100
        print(f"  {int(year)}年: 实际={actual:.0f}万, 预测={pred:.0f}万, 偏差={error:+.0f}万 ({error_pct:.2f}%)")

    # 5.8 未来预测
    future_years = np.arange(2024, 2031)
    pred_future = gm.predict(len(future_years))

    print("\n" + "=" * 60)
    print("未来人口预测 (2024-2030)")
    print("=" * 60)
    for year, pred in zip(future_years, pred_future):
        print(f"  {year}年: {pred:.0f} 万人 ({pred / 10000:.2f}亿)")

    # 5.9 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：拟合效果
    ax1 = axes[0]
    ax1.plot(years_train, x_train, 'o', label='训练数据', markersize=4, color='blue')
    ax1.plot(years_train, x_pred_train, '-', label='灰色模型拟合', linewidth=2, color='red')
    ax1.plot(years_valid, x_valid, 's', label='验证数据', markersize=6, color='green')
    ax1.plot(years_valid, x_pred_valid, '--', label='灰色模型预测', linewidth=2, color='orange')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('总人口（万人）')
    ax1.set_title('灰色预测模型 GM(1,1) 拟合效果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图：未来预测
    ax2 = axes[1]
    ax2.plot(years_train, x_train, 'o', label='历史数据', markersize=3, color='blue', alpha=0.5)
    ax2.plot(years_valid, x_valid, 's', label='验证数据', markersize=5, color='green')
    ax2.plot(future_years, pred_future, '--o', label='灰色模型预测', linewidth=2, color='red', markersize=4)
    ax2.axvline(x=2023, color='gray', linestyle='--', alpha=0.5, label='预测起点')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('总人口（万人）')
    ax2.set_title('2024-2030年人口预测')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 5.10 与原有模型对比
    print("\n" + "=" * 60)
    print("模型对比")
    print("=" * 60)

    comparison = pd.DataFrame({
        '模型': ['灰色预测 GM(1,1)', '指数模型', '改进指数模型', 'Logistic模型'],
        '验证集MAE(万人)': [valid_metrics['mae'], 26122, 1741, 1944],
        '验证集RMSE(万人)': [valid_metrics['rmse'], 26197, 1753, 2030],
        '2030年预测(万人)': [pred_future[-1], 186504, 138750, 147093]
    })

    print(comparison.to_string(index=False))