#数据读取与预处理：读取JHU原始数据，提取日本疫情数据，进行平滑和划分
# 读取原始数据 → 选择地区 → 计算新增 → 平滑处理 → 划分训练集/测试集 → 保存处理后的数据。
"""
数据预处理模块
读取JHU原始数据，提取日本疫情数据，进行平滑和划分
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')  # 忽略所有警告
# 设置matplotlib使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

DATA_DIR = 'E:/计算机建模技术大作业内容-202606/COVID-19-数据/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/'
OUTPUT_DIR = 'D:/Adaima/Computer_build_model/Finalhomework/result'

def load_jhu_data():
    """加载JHU三个核心数据文件"""
    confirmed_path = DATA_DIR + 'time_series_covid19_confirmed_global.csv'
    deaths_path = DATA_DIR + 'time_series_covid19_deaths_global.csv'
    recovered_path = DATA_DIR + 'time_series_covid19_recovered_global.csv'

    # 检查文件是否存在
    for path in [confirmed_path, deaths_path, recovered_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到文件: {path}")

    confirmed = pd.read_csv(confirmed_path)
    deaths = pd.read_csv(deaths_path)
    recovered = pd.read_csv(recovered_path)
    return confirmed, deaths, recovered

def extract_country_data(df, country_name):
    """提取指定国家的时序数据"""
    country_df = df[df['Country/Region'] == country_name]
    # 如果有多个省份，按列求和聚合
    if country_df.shape[0] > 1:
        series = country_df.iloc[:, 4:].sum(axis=0)
    else:
        series = country_df.iloc[0, 4:]
    return series

def preprocess_japan_data(confirmed, deaths, recovered):
    """处理日本数据，计算每日新增并平滑"""
    # 提取时间序列
    japan_conf = extract_country_data(confirmed, 'Japan')
    japan_death = extract_country_data(deaths, 'Japan')
    japan_rec = extract_country_data(recovered, 'Japan')

    # 构造DataFrame
    dates = pd.to_datetime(japan_conf.index, errors='coerce')
    df = pd.DataFrame({
        'date': dates,
        'confirmed': japan_conf.values,
        'deaths': japan_death.values,
        'recovered': japan_rec.values
    })

    # 删除无效日期
    df = df.dropna(subset=['date'])

    # 计算每日新增
    df['new_confirmed'] = df['confirmed'].diff().fillna(0)
    df['new_deaths'] = df['deaths'].diff().fillna(0)
    df['new_recovered'] = df['recovered'].diff().fillna(0)

    # 处理异常值（防止负数）
    df['new_confirmed'] = df['new_confirmed'].clip(lower=0)
    df['new_deaths'] = df['new_deaths'].clip(lower=0)
    df['new_recovered'] = df['new_recovered'].clip(lower=0)

    # 7天移动平均平滑
    df['new_confirmed_smooth'] = df['new_confirmed'].rolling(window=7, center=True).mean()
    df['new_deaths_smooth'] = df['new_deaths'].rolling(window=7, center=True).mean()
    df['new_recovered_smooth'] = df['new_recovered'].rolling(window=7, center=True).mean()

    # 去掉首尾NaN（由于rolling center=True导致）
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

def split_train_test(df, train_ratio=0.8):
    """划分训练集和测试集"""
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    return train_df, test_df

def save_processed_data(df, train_df, test_df, output_dir='D:/Adaima/Computer_build_model/Finalhomework/result/data/'):
    """保存处理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir + 'japan_covid_full.csv', index=False)
    train_df.to_csv(output_dir + 'japan_covid_train.csv', index=False)
    test_df.to_csv(output_dir + 'japan_covid_test.csv', index=False)
    print(f"数据已保存至 {output_dir}")

def main():
    print("=" * 50)
    print("开始数据预处理...")

    # 1. 加载数据
    confirmed, deaths, recovered = load_jhu_data()
    print("原始数据加载完成")

    # 2. 处理日本数据
    df_clean = preprocess_japan_data(confirmed, deaths, recovered)
    print(f"日本数据提取完成，共 {len(df_clean)} 天")

    # 3. 划分训练/测试
    train_df, test_df = split_train_test(df_clean, train_ratio=0.8)
    train_size = len(train_df)  # 关键修复：获取训练集大小
    print(f"训练集: {train_size} 天, 测试集: {len(test_df)} 天")

    # 4. 保存
    save_processed_data(df_clean, train_df, test_df)

    # 5. 快速可视化检查
    plt.figure(figsize=(12, 4))
    plt.plot(df_clean['date'], df_clean['new_confirmed'], alpha=0.5, label='原始每日新增')
    plt.plot(df_clean['date'], df_clean['new_confirmed_smooth'], 'r-', linewidth=2, label='7日移动平均')
    plt.axvline(x=df_clean['date'].iloc[train_size], color='k', linestyle='--', label='训练/测试分界')
    plt.legend()
    plt.title('日本每日新增确诊（原始 vs 平滑）')
    plt.tight_layout()
    plt.savefig('D:/Adaima/Computer_build_model/Finalhomework/result/data_overview.png', dpi=150)
    plt.show()

    print("数据预处理完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()


# 生成的图片data_overview展示的是日本从2020年1月到2023年5月的每日新增确诊COVID-19病例时间序列，
# 包含原始数据和7日移动平均平滑后的趋势
'''
图中各元素的含义
图例	含义	作用
蓝色散点（原始每日新增）	每天官方报告的新增确诊数	显示每日波动，有大量噪声
红色曲线（7日移动平均）	过去7天新增数的平均值	平滑掉周末/节假日报告波动，显示真实趋势
黑色虚线（训练/测试分界）	2022年9月左右	前80%数据用于训练模型，后20%用于验证预测效果

从图中可以看出日本的疫情经历了8波明显的爆发
波次	时间	峰值（约）	特点
第1波	2020年3-4月	~500	早期爆发，规模较小
第2波	2020年7-8月	~1,500	夏季反弹
第3波	2020年11-12月	~3,000	冬季爆发
第4波	2021年4-5月	~7,000	樱花季+变异株
第5波	2021年8月	~25,000	德尔塔变异株，奥运会期间
第6波	2022年1-2月	~100,000	奥密克戎BA.1
第7波	2022年7-8月	~250,000	奥密克戎BA.5
第8波	2022年12月-2023年1月	~220,000	冬季+新变异株

从图中可见，日本的疫情经历了多个明显的爆发波次，其中第7波（2022年8月）达到峰值约25万例/日，是疫情以来的最高点。此后2023年初仍有一次较小的反弹。

训练集数据截止到2022年9月（黑色虚线），之后的数据作为测试集，用于评估模型对未知疫情趋势的预测能力。可以看到测试集包含了一段完整的疫情下降期和一次明显的反弹（2023年初），这将有效检验模型在复杂传播情景下的泛化能力。

'''