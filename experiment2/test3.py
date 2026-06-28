# ==================== 蠓虫分类实验 - 坐标交换对比 ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据准备 ====================
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

# 交换坐标后的数据（翅长, 触角长）
apf_swapped = apf_original[:, [1, 0]]  # 交换两列
af_swapped = af_original[:, [1, 0]]
unknown_swapped = unknown_original[:, [1, 0]]

# 合并数据和标签
X_orig = np.vstack([apf_original, af_original])
X_swap = np.vstack([apf_swapped, af_swapped])
y = np.array([1] * 6 + [0] * 9)  # 1: Apf(毒蠓), 0: Af(益蠓)

print("=" * 70)
print("蠓虫分类实验 - 坐标交换对比分析")
print("=" * 70)

# ==================== 2. 训练LDA模型 ====================
lda_orig = LinearDiscriminantAnalysis()
lda_orig.fit(X_orig, y)

lda_swap = LinearDiscriminantAnalysis()
lda_swap.fit(X_swap, y)

# 获取模型参数
print("\n【模型参数对比】")
print("-" * 70)
print(f"原始坐标系 (触角长=X, 翅长=Y):")
print(f"  Af均值: {lda_orig.means_[0]}")
print(f"  Apf均值: {lda_orig.means_[1]}")
print(f"  先验概率: {lda_orig.priors_}")
print(f"  决策边界系数 coef_: {lda_orig.coef_}")
print(f"  决策边界截距 intercept_: {lda_orig.intercept_}")

print(f"\n交换坐标系 (翅长=X, 触角长=Y):")
print(f"  Af均值: {lda_swap.means_[0]}")
print(f"  Apf均值: {lda_swap.means_[1]}")
print(f"  先验概率: {lda_swap.priors_}")
print(f"  决策边界系数 coef_: {lda_swap.coef_}")
print(f"  决策边界截距 intercept_: {lda_swap.intercept_}")


# ==================== 3. 计算决策边界方程 ====================
def get_decision_boundary_from_sklearn(lda_model, is_swapped=False):
    """
    使用sklearn LDA模型的 coef_ 和 intercept_ 获取决策边界
    sklearn的决策边界公式: coef_[0]*x1 + coef_[1]*x2 + intercept_ = 0
    即: coef_[0]*x + coef_[1]*y + intercept_ = 0
    解出 y = (-coef_[0]*x - intercept_) / coef_[1]
    """
    coef = lda_model.coef_[0]  # 系数数组
    intercept = lda_model.intercept_[0]  # 截距

    # 决策边界方程: coef[0]*x + coef[1]*y + intercept = 0
    # 解出: y = (-coef[0]*x - intercept) / coef[1]
    if abs(coef[1]) > 1e-6:
        slope = -coef[0] / coef[1]
        intercept_y = -intercept / coef[1]
    else:
        slope = np.inf
        intercept_y = -intercept / coef[0] if coef[0] != 0 else 0

    return slope, intercept_y, coef, intercept


# 计算原始坐标系的决策边界
slope_orig, intercept_orig, coef_orig, intercept_const_orig = get_decision_boundary_from_sklearn(lda_orig)
print(f"\n【决策边界方程】")
print("-" * 70)
print(f"原始坐标系: 翅长 = {slope_orig:.4f} × 触角长 + {intercept_orig:.4f}")
print(f"  原始方程: {coef_orig[0]:.4f}×触角长 + {coef_orig[1]:.4f}×翅长 + {intercept_const_orig:.4f} = 0")

# 计算交换坐标系的决策边界
slope_swap, intercept_swap, coef_swap, intercept_const_swap = get_decision_boundary_from_sklearn(lda_swap)
print(f"\n交换坐标系: 触角长 = {slope_swap:.4f} × 翅长 + {intercept_swap:.4f}")
print(f"  原始方程: {coef_swap[0]:.4f}×翅长 + {coef_swap[1]:.4f}×触角长 + {intercept_const_swap:.4f} = 0")


# ==================== 4. 计算两条边界线的交点 ====================
def find_intersection_point():
    """
    计算原始坐标系中两条决策边界的交点
    线1（原始LDA边界）: y = slope_orig * x + intercept_orig
    线2（交换坐标边界在原始坐标系中的表示）: y = (x - intercept_swap) / slope_swap
    """
    if abs(slope_swap) < 1e-6:
        return None, None, "交换坐标边界为水平线（斜率0），无法计算交点"

    # 交换坐标边界在原始坐标系中的表示
    # 原边界: 触角长 = slope_swap × 翅长 + intercept_swap
    # 即: x = slope_swap * y + intercept_swap
    # 解出: y = (x - intercept_swap) / slope_swap
    slope2 = 1 / slope_swap
    intercept2 = -intercept_swap / slope_swap

    # 解方程组
    # y = slope_orig * x + intercept_orig
    # y = slope2 * x + intercept2
    if abs(slope_orig - slope2) < 1e-6:
        return None, None, "两条直线平行，无交点"

    x_intersect = (intercept2 - intercept_orig) / (slope_orig - slope2)
    y_intersect = slope_orig * x_intersect + intercept_orig

    return x_intersect, y_intersect, "成功"


intersect_x, intersect_y, status = find_intersection_point()

print(f"\n【两条决策边界的交点】")
print("-" * 70)
if intersect_x is not None and intersect_y is not None:
    print(f"交点坐标: (触角长={intersect_x:.4f}, 翅长={intersect_y:.4f})")

    # 计算两组均值的中心点
    mu_apf = lda_orig.means_[1]
    mu_af = lda_orig.means_[0]
    center_x = (mu_apf[0] + mu_af[0]) / 2
    center_y = (mu_apf[1] + mu_af[1]) / 2
    print(f"两类均值中心点: ({center_x:.4f}, {center_y:.4f})")
    print(f"交点与中心点距离: {np.sqrt((intersect_x - center_x) ** 2 + (intersect_y - center_y) ** 2):.6f}")
else:
    print(f"无法计算交点: {status}")

# ==================== 5. 待判样本预测对比 ====================
print(f"\n【待判样本预测结果对比】")
print("-" * 70)
print("样本\t原始坐标系(触角长,翅长)\t\t交换坐标系(翅长,触角长)")
print("\tLDA预测\tP(Apf)\t\tLDA预测\tP(Apf)")
print("-" * 70)

for i in range(len(unknown_original)):
    pred_orig = lda_orig.predict([unknown_original[i]])[0]
    proba_orig = lda_orig.predict_proba([unknown_original[i]])[0, 1]
    pred_swap = lda_swap.predict([unknown_swapped[i]])[0]
    proba_swap = lda_swap.predict_proba([unknown_swapped[i]])[0, 1]

    pred_orig_label = "Apf(毒蠓)" if pred_orig == 1 else "Af(益蠓)"
    pred_swap_label = "Apf(毒蠓)" if pred_swap == 1 else "Af(益蠓)"

    print(f"{i + 1}\t{pred_orig_label}\t{proba_orig:.4f}\t\t{pred_swap_label}\t{proba_swap:.4f}")

# ==================== 6. 可视化对比 ====================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ===== 左图：原始坐标系中的两条决策边界 =====
ax1 = axes[0]

# 绘制训练数据
ax1.scatter(apf_original[:, 0], apf_original[:, 1], c='red', s=80, marker='o',
            label='Apf(毒蠓)', edgecolors='black', linewidth=1.5)
ax1.scatter(af_original[:, 0], af_original[:, 1], c='green', s=80, marker='s',
            label='Af(益蠓)', edgecolors='black', linewidth=1.5)
ax1.scatter(unknown_original[:, 0], unknown_original[:, 1], c='blue', s=120, marker='^',
            label='待判样本', edgecolors='black', linewidth=1.5)

# 绘制原始LDA决策边界
x_range = np.array([1.10, 1.60])
if np.isfinite(slope_orig):
    y_boundary_orig = slope_orig * x_range + intercept_orig
    ax1.plot(x_range, y_boundary_orig, 'k-', linewidth=2, label=f'原始LDA边界')

# 绘制交换坐标系的LDA决策边界（变换回原始坐标系）
if np.isfinite(slope_swap) and abs(slope_swap) > 1e-6:
    # 交换坐标边界在原始坐标系中的表示: y = (x - intercept_swap) / slope_swap
    y_boundary_swapped = (x_range - intercept_swap) / slope_swap
    ax1.plot(x_range, y_boundary_swapped, 'k--', linewidth=2,
             label=f'交换坐标边界(变换后)')

# 标记交点
if intersect_x is not None and intersect_y is not None:
    ax1.scatter(intersect_x, intersect_y, c='purple', s=150, marker='X',
                zorder=10, label=f'边界交点({intersect_x:.3f},{intersect_y:.3f})')

# 标记两类均值中心点
mu_apf = lda_orig.means_[1]
mu_af = lda_orig.means_[0]
center_x = (mu_apf[0] + mu_af[0]) / 2
center_y = (mu_apf[1] + mu_af[1]) / 2
ax1.scatter(center_x, center_y, c='orange', s=100, marker='+',
            linewidths=2, zorder=10, label='均值中心点')

ax1.set_xlabel('触角长 (mm)', fontsize=12)
ax1.set_ylabel('翅长 (mm)', fontsize=12)
ax1.set_title('原始坐标系中的决策边界对比', fontsize=14)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1.10, 1.65)
ax1.set_ylim(1.60, 2.15)

# ===== 右图：交换坐标系中的两条决策边界 =====
ax2 = axes[1]

# 绘制交换坐标系中的训练数据
ax2.scatter(apf_swapped[:, 0], apf_swapped[:, 1], c='red', s=80, marker='o',
            label='Apf(毒蠓)', edgecolors='black', linewidth=1.5)
ax2.scatter(af_swapped[:, 0], af_swapped[:, 1], c='green', s=80, marker='s',
            label='Af(益蠓)', edgecolors='black', linewidth=1.5)
ax2.scatter(unknown_swapped[:, 0], unknown_swapped[:, 1], c='blue', s=120, marker='^',
            label='待判样本', edgecolors='black', linewidth=1.5)

# 绘制交换坐标系的LDA决策边界
x_range_swap = np.array([1.60, 2.15])
if np.isfinite(slope_swap):
    y_boundary_swap = slope_swap * x_range_swap + intercept_swap
    ax2.plot(x_range_swap, y_boundary_swap, 'k-', linewidth=2, label='交换坐标LDA边界')

# 绘制原始LDA边界（变换到交换坐标系）
if np.isfinite(slope_orig):
    # 原始边界在交换坐标系中: Y = slope_orig × X + intercept_orig (但X=翅长, Y=触角长)
    y_boundary_orig_swapped = slope_orig * x_range_swap + intercept_orig
    ax2.plot(x_range_swap, y_boundary_orig_swapped, 'k--', linewidth=2,
             label='原始LDA边界(变换后)')

# 标记均值中心点（在交换坐标系中）
mu_apf_swap = lda_swap.means_[1]
mu_af_swap = lda_swap.means_[0]
center_x_swap = (mu_apf_swap[0] + mu_af_swap[0]) / 2
center_y_swap = (mu_apf_swap[1] + mu_af_swap[1]) / 2
ax2.scatter(center_x_swap, center_y_swap, c='orange', s=100, marker='+',
            linewidths=2, zorder=10, label='均值中心点')

ax2.set_xlabel('翅长 (mm)', fontsize=12)
ax2.set_ylabel('触角长 (mm)', fontsize=12)
ax2.set_title('交换坐标系中的决策边界对比', fontsize=14)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1.60, 2.15)
ax2.set_ylim(1.10, 1.65)

plt.tight_layout()
plt.savefig('坐标交换对比图_带交点.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 7. 总结输出 ====================
print("\n" + "=" * 70)
print("结论总结")
print("=" * 70)
print("""
1. 【决策边界交点】
   两条决策边界相交于一点。该点理论上位于两类均值连线的中垂线上。

2. 【交换坐标的影响】
   - 决策边界的斜率和截距会改变
   - 但两条边界线会相交于同一点
   - 分类结果可能不同

3. 【实际意义】
   - 特征选择会影响分类结果
   - 不能随意交换特征，需要根据实际物理意义选择
""")