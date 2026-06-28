import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据（触角长, 翅长）
apf_original = np.array([
    [1.14, 1.78], [1.18, 1.96], [1.20, 1.86],
    [1.26, 2.00], [1.28, 2.00], [1.30, 1.96]
])
af_original = np.array([
    [1.24, 1.72], [1.36, 1.74], [1.38, 1.64],
    [1.38, 1.82], [1.38, 1.90], [1.40, 1.70],
    [1.48, 1.82], [1.54, 1.82], [1.56, 2.08]
])
unknown_original = np.array([
    [1.24, 1.80], [1.28, 1.84], [1.40, 2.04]
])

# 交换坐标（翅长, 触角长）
apf_swapped = apf_original[:, [1, 0]]  # 交换两列
af_swapped = af_original[:, [1, 0]]
unknown_swapped = unknown_original[:, [1, 0]]

# 训练原始模型
X_orig = np.vstack([apf_original, af_original])
y = np.array([1]*6 + [0]*9)
lda_orig = LinearDiscriminantAnalysis()
lda_orig.fit(X_orig, y)

# 训练交换坐标模型
X_swap = np.vstack([apf_swapped, af_swapped])
lda_swap = LinearDiscriminantAnalysis()
lda_swap.fit(X_swap, y)

# 预测
print("="*60)
print("原始坐标预测结果:")
for i, x in enumerate(unknown_original):
    pred = lda_orig.predict([x])[0]
    proba = lda_orig.predict_proba([x])[0, 1]
    print(f"  样本{i+1}: 预测={'Apf' if pred==1 else 'Af'}, P(Apf)={proba:.4f}")

print("\n交换坐标后预测结果:")
for i, x in enumerate(unknown_swapped):
    pred = lda_swap.predict([x])[0]
    proba = lda_swap.predict_proba([x])[0, 1]
    print(f"  样本{i+1}: 预测={'Apf' if pred==1 else 'Af'}, P(Apf)={proba:.4f}")

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：原始坐标
ax1 = axes[0]
ax1.scatter(apf_original[:,0], apf_original[:,1], c='red', s=80, label='Apf')
ax1.scatter(af_original[:,0], af_original[:,1], c='green', s=80, label='Af')
ax1.scatter(unknown_original[:,0], unknown_original[:,1], c='blue', s=120, marker='^')
ax1.set_xlabel('触角长 (mm)', fontsize=12)
ax1.set_ylabel('翅长 (mm)', fontsize=12)
ax1.set_title('原始坐标', fontsize=14)
ax1.legend()
ax1.axis('equal')

# 绘制决策边界
x_min, x_max = ax1.get_xlim()
y_min, y_max = ax1.get_ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = lda_orig.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax1.contour(xx, yy, Z, colors='black', linewidths=2)

# 右图：交换坐标
ax2 = axes[1]
ax2.scatter(apf_swapped[:,0], apf_swapped[:,1], c='red', s=80, label='Apf')
ax2.scatter(af_swapped[:,0], af_swapped[:,1], c='green', s=80, label='Af')
ax2.scatter(unknown_swapped[:,0], unknown_swapped[:,1], c='blue', s=120, marker='^')
ax2.set_xlabel('翅长 (mm)', fontsize=12)
ax2.set_ylabel('触角长 (mm)', fontsize=12)
ax2.set_title('交换坐标 (翅长=X, 触角长=Y)', fontsize=14)
ax2.legend()
ax2.axis('equal')

# 绘制决策边界
x_min, x_max = ax2.get_xlim()
y_min, y_max = ax2.get_ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = lda_swap.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax2.contour(xx, yy, Z, colors='black', linewidths=2)

plt.tight_layout()
plt.savefig('坐标交换对比图.png', dpi=150, bbox_inches='tight')
plt.show()