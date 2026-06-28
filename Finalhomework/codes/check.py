import pandas as pd
import numpy as np

seir = pd.read_csv('D:/Adaima/Computer_build_model/Finalhomework/result/data/seir_results.csv')
print("SEIR predicted_new 统计:")
print(f"  最小值: {seir['predicted_new'].min():.2f}")
print(f"  最大值: {seir['predicted_new'].max():.2f}")
print(f"  均值: {seir['predicted_new'].mean():.2f}")
print(f"  前10个值: {seir['predicted_new'].head(10).tolist()}")