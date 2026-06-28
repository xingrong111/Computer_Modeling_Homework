#构造LSTM时序数据 → 训练模型 → 多步预测（7天/14天）→ 保存预测结果。
"""
LSTM深度学习预测模块
包含数据构造、模型训练、多步预测
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示错误和警告，不显示info
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 关闭oneDNN加速信息

def create_sequences(data, seq_length=15):
    """创建时间序列样本"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(seq_length):
    """构建LSTM模型"""
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def forecast_future(model, last_seq, scaler, seq_length, days=7):
    """滚动预测未来多天"""
    current_seq = last_seq.copy().flatten()
    predictions = []

    for _ in range(days):
        # 预测下一天
        pred_scaled = model.predict(current_seq.reshape(1, seq_length, 1), verbose=0)
        predictions.append(pred_scaled[0, 0])
        # 更新序列
        current_seq = np.roll(current_seq, -1)
        current_seq[-1] = pred_scaled[0, 0]

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


def main():
    print("=" * 50)
    print("LSTM模型训练与预测...")

    # 加载数据
    full_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_full.csv')
    train_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_train.csv')
    test_df = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/japan_covid_test.csv')

    # 使用平滑后的每日新增数据
    data = full_df['new_confirmed_smooth'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # 划分训练/测试（按索引）
    train_size = len(train_df)
    train_data = data_scaled[:train_size]
    test_data = data_scaled[train_size:]

    # 创建序列
    SEQ_LENGTH = 15
    X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
    X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

    # 重塑为LSTM输入格式
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    print(f"训练样本: {X_train.shape[0]}, 测试样本: {X_test.shape[0]}")

    # 构建并训练模型
    model = build_lstm_model(SEQ_LENGTH)
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    import pickle
    with open('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("LSTM训练历史已保存")
    # 测试集评估
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)
    print(f"LSTM测试集 - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")

    # 保存预测结果
    lstm_results = pd.DataFrame({
        'date': test_df['date'].iloc[SEQ_LENGTH:].reset_index(drop=True),
        'actual': y_test_actual.flatten(),
        'predicted': y_pred.flatten()
    })
    lstm_results.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_results.csv', index=False)

    # 多步预测：7天和14天
    last_seq = data_scaled[-SEQ_LENGTH:].flatten()
    pred_7d = forecast_future(model, last_seq, scaler, SEQ_LENGTH, days=7)
    pred_14d = forecast_future(model, last_seq, scaler, SEQ_LENGTH, days=14)

    # 构造未来日期
    last_date = pd.to_datetime(full_df['date'].iloc[-1])
    future_dates_7 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
    future_dates_14 = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)

    # 方案1：分别保存两个预测结果（推荐）
    forecast_7d_df = pd.DataFrame({
        'date': future_dates_7,
        'predicted': pred_7d
    })
    forecast_7d_df.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_7d.csv', index=False)

    forecast_14d_df = pd.DataFrame({
        'date': future_dates_14,
        'predicted': pred_14d
    })
    forecast_14d_df.to_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/lstm_forecast_14d.csv', index=False)

    print(f"未来7天预测: {pred_7d.round(0)}")
    print(f"未来14天预测: {pred_14d.round(0)}")
    print("=" * 50)
if __name__ == "__main__":
    main()

'''
LSTM测试集 - RMSE: 14431.83, MAE: 8776.74, R²: 0.9427
未来7天预测: [11327. 11316. 11389. 11602. 11934. 12359. 12836.]
未来14天预测: [11327. 11316. 11389. 11602. 11934. 12359. 12836. 13359. 13922. 14590.
 15265. 15948. 16659. 17382.]
未来7天预测: [11327. 11316. 11389. 11602. 11934. 12359. 12836.]
未来14天预测: [11327. 11316. 11389. 11602. 11934. 12359. 12836. 13359. 13922. 14590.
 15265. 15948. 16659. 17382.]
==================================================

设置了早停机制，模型大概在第18轮左右已经达到了最佳，继续训练只会过拟合

'''
