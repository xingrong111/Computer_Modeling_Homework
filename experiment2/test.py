# ==================== 蠓虫分类实验完整代码====================
# 实验二：蠓虫分类（LDA / QDA / 代价敏感分类）
#数据准备  模型训练   模型评估   待判样本识别   判别边界可视化  代价敏感分类  总结输出
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, LeaveOneOut

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据准备 ====================
# 蠓虫数据  （触角长、翅长）mm  apf = 1是有害  apf = 0是有益的
# Apf（毒蠓，有害） - 6个样本
apf = np.array([
    [1.14, 1.78],
    [1.18, 1.96],
    [1.20, 1.86],
    [1.26, 2.00],
    [1.28, 2.00],
    [1.30, 1.96]
])

# Af（益蠓，有益） - 9个样本
af = np.array([
    [1.24, 1.72],
    [1.36, 1.74],
    [1.38, 1.64],
    [1.38, 1.82],
    [1.38, 1.90],
    [1.40, 1.70],
    [1.48, 1.82],
    [1.54, 1.82],
    [1.56, 2.08]
])

# 待判样本（触角长，翅长） - 3个
unknown = np.array([
    [1.24, 1.80],
    [1.28, 1.84],
    [1.40, 2.04]
])

# 合并数据并创建标签
#X是特征矩阵，Y是标签向量
X = np.vstack([apf, af])   #垂直堆叠数组，将apf和af按行进行拼接
y = np.array([1] * len(apf) + [0] * len(af))  # 1: Apf(有害), 0: Af(有益)
#分别对应创建6个1和9个0的标签，分别表示毒蠓和益蠓，最后合并标签得到y

# 转换为DataFrame
df_apf = pd.DataFrame(apf, columns=['触角长', '翅长'])
df_apf['类别'] = 'Apf(毒蠓)'
df_af = pd.DataFrame(af, columns=['触角长', '翅长'])
df_af['类别'] = 'Af(益蠓)'
df_all = pd.concat([df_apf, df_af], ignore_index=True)

print("=" * 60)
print("蠓虫分类实验数据")
print("=" * 60)
print("\nApf(毒蠓)样本 (6个):")
print(df_apf)
print("\nAf(益蠓)样本 (9个):")
print(df_af)
print("\n待判样本 (3个):")
print(pd.DataFrame(unknown, columns=['触角长', '翅长']))

# ==================== 2. 数据可视化 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：样本分布散点图，展示两类样本分布，观察是否线性可分
#apf（红色圆圈）集中在左上方，af（绿色方块）分布较分散，待判样本（蓝色三角形）位于中间区域，两类样本存在重叠，不是完全可分的
ax1 = axes[0]
ax1.scatter(apf[:, 0], apf[:, 1], color='red', s=80, marker='o',
            label='Apf (毒蠓, 有害)', edgecolors='black', linewidth=1.5)
ax1.scatter(af[:, 0], af[:, 1], color='green', s=80, marker='s',
            label='Af (益蠓, 有益)', edgecolors='black', linewidth=1.5)
ax1.scatter(unknown[:, 0], unknown[:, 1], color='blue', s=120, marker='^',
            label='待判样本', edgecolors='black', linewidth=1.5)
ax1.set_xlabel('触角长 (mm)', fontsize=12)
ax1.set_ylabel('翅长 (mm)', fontsize=12)
ax1.set_title('蠓虫样本分布', fontsize=14)
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# 子图2：箱线图 - 特征对比，对比两类蠓虫在触角长、翅长上的差异
# Af触角长箱线图：中位数约1.40，范围1.24-1.56，分布较宽
# Af翅长箱线图：中位数约1.82，范围1.64-2.08
# Apf触角长箱线图：中位数约1.26，范围1.14-1.30，分布较窄
# Apf翅长箱线图：中位数约1.96，范围1.78-2.00
# 关键发现：Apf触角长普遍小于Af，但翅长普遍大于Af
ax2 = axes[1]
data_to_plot = []
tick_labels = []
for label, name in [(0, 'Af(益蠓)'), (1, 'Apf(毒蠓)')]:
    data_to_plot.append(X[y == label, 0])
    tick_labels.append(f'{name}\n触角长')
    data_to_plot.append(X[y == label, 1])
    tick_labels.append(f'{name}\n翅长')
bp = ax2.boxplot(data_to_plot, tick_labels=tick_labels, patch_artist=True,
                 boxprops=dict(linewidth=1.5), medianprops=dict(linewidth=2))
colors = ['lightgreen', 'lightgreen', 'lightcoral', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_ylabel('长度 (mm)', fontsize=12)
ax2.set_title('两类蠓虫特征对比', fontsize=14)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('蠓虫数据分布图.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 3. 模型训练 ====================
print("\n" + "=" * 60)
print("模型训练与评估")
print("=" * 60)

# 1. 线性判别分析 (LDA)：假设各类协方差矩阵相同
#lda = LinearDiscriminantAnalysis()  #创建LDA分类器实例，初始化一个LDA模型对象
lda = LinearDiscriminantAnalysis(priors=[0.8, 0.2])  #创建LDA分类器实例，初始化一个LDA模型对象
lda.fit(X, y)  #训练模型，激素按类均值、共同协方差、先验概率
y_pred_lda = lda.predict(X)   #预测类别，根据判别函数输出类别  0 / 1
y_pred_proba_lda = lda.predict_proba(X)  #预测概率  输出P(af) P(apf)和为1

# 2 二次判别分析 (QDA)：允许各类协方差不同
#qda = QuadraticDiscriminantAnalysis()  #QDA允许各类协方差矩阵不同
qda = QuadraticDiscriminantAnalysis(priors=[0.5, 0.5])  #QDA允许各类协方差矩阵不同
qda.fit(X, y)  #对每个类别单独计算协方差
y_pred_qda = qda.predict(X)
y_pred_proba_qda = qda.predict_proba(X)

# ==================== 4. 模型评估 ====================
# 4.1 混淆矩阵和分类报告
#混淆矩阵，  2*2  行+真实值，列=预测值
# TN（真阴性）	真实0，预测0	Af正确判为Af
# FP（假阳性）	真实0，预测1	Af错误判为Apf
# FN（假阴性）	真实1，预测0	Apf错误判为Af
# TP（真阳性）	真实1，预测1	Apf正确判为Apf
print("\n【线性判别分析 LDA】")
print("混淆矩阵:")
cm_lda = confusion_matrix(y, y_pred_lda)
print(cm_lda)
#得到
# 第一行[9, 0]：9个Af样本，0个被误判
# 第二行[0, 6]：6个Apf样本，0个被误判
# 结论：LDA在训练集上完美分类，准确率100%
print("\n分类报告:")
#精确率P：TP/(TP+FP)  预测为Positive中实际为Positive的比例
#召回率R：TP/(TP+FN)  实际为Positive中被正确预测的比例
#F1分数：2×P×R/(P+R)  精确率和召回率的调和平均
#准确率A：(TP+TN)/(TP+TN+FP+FN)	总体预测正确的比例
print(classification_report(y, y_pred_lda, target_names=['Af(益蠓)', 'Apf(毒蠓)']))
print(f"准确率: {accuracy_score(y, y_pred_lda):.4f}")
#输出：
# 所有指标均为1.00 = 完美分类
# support：该类别的实际样本数

print("\n【二次判别分析 QDA】")
print("混淆矩阵:")
cm_qda = confusion_matrix(y, y_pred_qda)
print(cm_qda)
print("\n分类报告:")
print(classification_report(y, y_pred_qda, target_names=['Af(益蠓)', 'Apf(毒蠓)']))
print(f"准确率: {accuracy_score(y, y_pred_qda):.4f}")

# 交叉验证 (留一法)：小样本下更可靠的评估 ———— 每次留1个样本作测试，其余训练，实验中一共进行15次训练和测试
loo = LeaveOneOut()
lda_cv_scores = cross_val_score(lda, X, y, cv=loo)
qda_cv_scores = cross_val_score(qda, X, y, cv=loo)
#cv = loo 使用留一法的交叉验证策略    cross_val_score 交叉验证评分函数，返回每次测试的准确率数组
print(f"\n留一法交叉验证结果:")
print(f"LDA 平均准确率: {lda_cv_scores.mean():.4f} ± {lda_cv_scores.std():.4f}")
print(f"QDA 平均准确率: {qda_cv_scores.mean():.4f} ± {qda_cv_scores.std():.4f}")
#mean()平均值，返回15次测试的平均准确率     std()标准差，评估预测的稳定性
#输出：
# LDA：平均93.33%准确率，标准差0.2494较大，说明存在某个样本被错误分类
# QDA：平均100%准确率，标准差0，说明15次测试全部正确
# 结论：QDA的泛化能力更强


# ==================== 5. 待判样本识别 ====================
#使用训练好的模型对3个未知样本进行分类，输出分类结果和属于Apf的概率
print("\n" + "=" * 60)
print("待判样本识别结果")
print("=" * 60)

unknown_pred_lda = lda.predict(unknown) #预测类别 1 / 0
unknown_proba_lda = lda.predict_proba(unknown)    #预测概率
unknown_pred_qda = qda.predict(unknown)
unknown_proba_qda = qda.predict_proba(unknown)

results_df = pd.DataFrame({
    '样本编号': [1, 2, 3],
    '触角长': unknown[:, 0],
    '翅长': unknown[:, 1],
    'LDA预测': ['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in unknown_pred_lda],
    'LDA-Apf概率': unknown_proba_lda[:, 1],
    'QDA预测': ['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in unknown_pred_qda],
    'QDA-Apf概率': unknown_proba_qda[:, 1]
})
print(results_df.to_string(index=False))
#输出：
# LDA认为三个样本都更可能是Apf（概率>72%）
# QDA认为三个样本都更可能是Af（概率>55%，即Apf概率<45%）
# 两个模型给出完全相反的结论


# ==================== 6. 判别边界可视化 ====================
#在特征空间绘制决策边界，直观展示LDA(直线边界)和QDA(曲线/二次边界)的区别
def plot_decision_boundary(X, y, model, title, ax):
    """绘制判别边界"""
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策区域
    ax.contourf(xx, yy, Z, alpha=0.3, colors=['green', 'red'])

    # 绘制原始样本
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='green', s=60, marker='s',
               label='Af(益蠓)', edgecolors='black', linewidth=1.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='red', s=60, marker='o',
               label='Apf(毒蠓)', edgecolors='black', linewidth=1.5)

    ax.set_xlabel('触角长 (mm)', fontsize=11)
    ax.set_ylabel('翅长 (mm)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_boundary(X, y, lda, 'LDA 线性判别边界', axes[0])
plot_decision_boundary(X, y, qda, 'QDA 二次判别边界', axes[1])
plt.tight_layout()
plt.savefig('判别边界图.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 7. 待判样本可视化 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (ax, model, name) in enumerate(zip(axes, [lda, qda], ['LDA', 'QDA'])):
    # 绘制背景区域
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, colors=['green', 'red'])

    # 绘制已知样本
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color='green', s=60, marker='s',
               label='Af(益蠓)', edgecolors='black', linewidth=1.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color='red', s=60, marker='o',
               label='Apf(毒蠓)', edgecolors='black', linewidth=1.5)

    # 绘制待判样本
    preds = model.predict(unknown)
    colors_u = ['green' if p == 0 else 'red' for p in preds]
    for i, (x, y_coord, c) in enumerate(zip(unknown[:, 0], unknown[:, 1], colors_u)):
        ax.scatter(x, y_coord, color=c, s=150, marker='^',
                   edgecolors='black', linewidth=2, zorder=5)
        ax.annotate(f'  #{i + 1}', (x, y_coord), fontsize=10, fontweight='bold')

    ax.set_xlabel('触角长 (mm)', fontsize=11)
    ax.set_ylabel('翅长 (mm)', fontsize=11)
    ax.set_title(f'{name} - 待判样本分类结果', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('待判样本分类图.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 8. 代价敏感分类 ====================
print("\n" + "=" * 60)
print("问题3: 代价敏感分类")
print("=" * 60)
print("背景: Apf(毒蠓)是疾病载体，Af(益蠓)是传粉益虫")
print("误判代价: 将毒蠓误判为益蠓 (假阴性) 代价更高")

# 方法1: 调整决策阈值
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]   #决策阈值，判定为Apf的概率门槛，默认为0.5，可调整
print("\n【方法1: 调整LDA决策阈值】")
print("阈值\t预测结果\t\t代价说明")
print("-" * 50)

for thresh in thresholds:
    cost_sensitive_pred = (lda.predict_proba(unknown)[:, 1] >= thresh).astype(int)
    pred_labels = ['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in cost_sensitive_pred]
    print(f"{thresh}\t{pred_labels}\t", end="")
    if thresh < 0.5:
        print("更保守 (更易判为毒蠓，避免漏判)")
    elif thresh > 0.5:
        print("更激进 (更易判为益蠓)")
    else:
        print("默认阈值")

# 推荐阈值: 0.3 (降低阈值，更倾向于判为毒蠓)
optimal_threshold = 0.3
cost_sensitive_pred = (lda.predict_proba(unknown)[:, 1] >= optimal_threshold).astype(int)
print(f"\n推荐使用阈值: {optimal_threshold}")
print(f"代价敏感分类结果: {['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in cost_sensitive_pred]}")

# 方法2: 使用自定义权重
print("\n【方法2: 类别加权】")
class_weight = {0: 1.0, 1: 5.0}  # Apf(毒蠓)权重更高
print(f"类别权重: Af(益蠓)={class_weight[0]}, Apf(毒蠓)={class_weight[1]}")
# print("(注: sklearn的LDA不支持直接类别加权，需自定义或使用其他算法)")

# ==================== 9. 总结输出====================
print("\n" + "=" * 60)
print("实验总结")
print("=" * 60)

# 使用正确的数值
summary = {
    '模型': ['LDA', 'QDA'],
    '训练准确率': [accuracy_score(y, y_pred_lda), accuracy_score(y, y_pred_qda)],
    '留一法CV准确率': [lda_cv_scores.mean(), qda_cv_scores.mean()],
    'Af→Apf误判数': [int(cm_lda[0, 1]), int(cm_qda[0, 1])],
    'Apf→Af误判数': [int(cm_lda[1, 0]), int(cm_qda[1, 0])]
}

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

print("\n待判样本最终结论:")
for i, (pred_lda, pred_qda) in enumerate(zip(unknown_pred_lda, unknown_pred_qda)):
    print(f"样本{i + 1}: LDA预测={'Apf(毒蠓)' if pred_lda == 1 else 'Af(益蠓)'}, "
          f"QDA预测={'Apf(毒蠓)' if pred_qda == 1 else 'Af(益蠓)'}")

print("\n代价敏感分类建议:")
print("为避免毒蠓被误判为益蠓造成疾病传播，建议采用阈值0.3的LDA模型")
print("此时3个待判样本均被判定为Apf(毒蠓)，采取更严格的防控措施")

# ==================== 8. 问题3: 修改分类判别方法 ====================
print("\n" + "=" * 80)
print("问题3: 修改分类判别方法（考虑实际代价）")
print("=" * 80)
print("背景说明:")
print("  - Apf(毒蠓): 疾病载体 → 误判为益蠓的代价极高（可能造成疾病传播）")
print("  - Af(益蠓): 传粉益虫 → 误判为毒蠓的代价较低（仅造成资源浪费）")
print("=" * 80)

# ========== 方法一：调整决策阈值 ==========
print("\n【修改方法1】调整决策阈值（更保守策略）")
print("-" * 60)

# 原始默认阈值0.5的LDA预测结果
original_lda_pred = lda.predict(unknown)
original_lda_proba = lda.predict_proba(unknown)

print("原始方法（默认阈值=0.5）:")
for i, (proba, pred) in enumerate(zip(original_lda_proba[:, 1], original_lda_pred)):
    print(f"  样本{i + 1}: P(Apf)={proba:.4f} → 预测={('Af(益蠓)' if pred == 0 else 'Apf(毒蠓)')}")

print("\n修改方法（降低阈值，更易判为毒蠓）:")

# 尝试不同阈值
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

for thresh in thresholds:
    modified_pred = (lda.predict_proba(unknown)[:, 1] >= thresh).astype(int)
    modified_labels = ['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in modified_pred]
    results.append(modified_labels)

    # 策略说明
    if thresh < 0.5:
        strategy = "保守策略（优先判为毒蠓，避免漏判）"
    elif thresh > 0.5:
        strategy = "激进策略（优先判为益蠓）"
    else:
        strategy = "默认策略"

    print(f"  阈值={thresh}: {modified_labels} ← {strategy}")

# 推荐阈值
recommended_thresh = 0.3
final_pred = (lda.predict_proba(unknown)[:, 1] >= recommended_thresh).astype(int)
print(f"\n★ 推荐方案: 采用阈值={recommended_thresh}的保守策略")
print(f"   识别结果: {['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in final_pred]}")

# ========== 方法二：代价矩阵法 ==========
print("\n" + "=" * 60)
print("\n【修改方法2】引入代价矩阵（最小化期望代价）")
print("-" * 60)

# 定义代价矩阵
# 行: 真实类别 (0=Af, 1=Apf)
# 列: 预测类别 (0=Af, 1=Apf)
cost_matrix = np.array([
    [0, 1],  # 真实Af: 判Af代价0, 判Apf代价1
    [10, 0]  # 真实Apf: 判Af代价10(高), 判Apf代价0
])

print("代价矩阵 (真实类别 × 预测类别):")
print("                 预测为Af  预测为Apf")
print(f"真实为Af(益蠓)    {cost_matrix[0, 0]}         {cost_matrix[0, 1]}")
print(f"真实为Apf(毒蠓)   {cost_matrix[1, 0]}         {cost_matrix[1, 1]}")
print("\n说明: 将毒蠓(Apf)误判为益蠓(Af)的代价是10，远高于将益蠓误判为毒蠓的代价1")


# 计算每个样本的期望代价
def expected_cost(proba_apf, cost_matrix):
    """计算判为各类别的期望代价"""
    proba_af = 1 - proba_apf
    # 判为Af的期望代价 = P(真实=Af)*代价(Af判Af) + P(真实=Apf)*代价(Apf判Af)
    cost_if_predict_af = proba_af * cost_matrix[0, 0] + proba_apf * cost_matrix[1, 0]
    # 判为Apf的期望代价 = P(真实=Af)*代价(Af判Apf) + P(真实=Apf)*代价(Apf判Apf)
    cost_if_predict_apf = proba_af * cost_matrix[0, 1] + proba_apf * cost_matrix[1, 1]
    return cost_if_predict_af, cost_if_predict_apf


print("\n基于期望代价最小化的决策:")
print("-" * 60)
print("样本\tP(Apf)\t判为Af的代价\t判为Apf的代价\t最优决策")
print("-" * 60)

cost_based_pred = []
for i, proba in enumerate(original_lda_proba[:, 1]):
    cost_af, cost_apf = expected_cost(proba, cost_matrix)
    # 选择期望代价更小的类别
    best_pred = 0 if cost_af < cost_apf else 1
    cost_based_pred.append(best_pred)
    best_label = "Af(益蠓)" if best_pred == 0 else "Apf(毒蠓)"
    print(f"{i + 1}\t{proba:.4f}\t{cost_af:.4f}\t\t{cost_apf:.4f}\t\t{best_label}")

print(f"\n★ 代价矩阵法识别结果: {['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in cost_based_pred]}")

# ========== 方法三：类别加权法 ==========
print("\n" + "=" * 60)
print("\n【修改方法3】类别加权法（使用class_weight参数）")
print("-" * 60)
print("原理: 给毒蠓(Apf)更高的权重，使模型更关注减少毒蠓的误判")

# 模拟加权：调整先验概率
# 原始先验概率（基于样本比例）
original_prior_apf = len(apf) / len(X)  # 6/15 = 0.4
original_prior_af = len(af) / len(X)  # 9/15 = 0.6

# 加权后的先验概率（提高Apf权重）
weight_apf = 3.0  # 给Apf的权重提高3倍
weight_af = 1.0
weighted_prior_apf = (len(apf) * weight_apf) / (len(apf) * weight_apf + len(af) * weight_af)
weighted_prior_af = 1 - weighted_prior_apf

print(f"\n原始先验概率: P(Apf)={original_prior_apf:.3f}, P(Af)={original_prior_af:.3f}")
print(f"加权后先验概率: P(Apf)={weighted_prior_apf:.3f}, P(Af)={weighted_prior_af:.3f}")
print(f"权重设置: Apf(毒蠓)权重={weight_apf}, Af(益蠓)权重={weight_af}")

# 使用加权先验重新训练LDA
lda_weighted = LinearDiscriminantAnalysis(priors=[weighted_prior_af, weighted_prior_apf])
lda_weighted.fit(X, y)
weighted_pred = lda_weighted.predict(unknown)
weighted_proba = lda_weighted.predict_proba(unknown)

print("\n加权LDA模型预测结果:")
for i, (proba, pred) in enumerate(zip(weighted_proba[:, 1], weighted_pred)):
    print(f"  样本{i + 1}: P(Apf)={proba:.4f} → 预测={'Af(益蠓)' if pred == 0 else 'Apf(毒蠓)'}")

print(f"\n★ 类别加权法识别结果: {['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in weighted_pred]}")

# ========== 方法四：自定义损失函数的分类器 ==========
print("\n" + "=" * 60)
print("\n【修改方法4】自定义分类器（基于贝叶斯决策理论）")
print("-" * 60)


def custom_bayes_classifier(proba_apf, cost_af_to_apf=1, cost_apf_to_af=10):
    """
    基于贝叶斯最小风险决策的自定义分类器

    参数:
    - proba_apf: 样本属于Apf的概率
    - cost_af_to_apf: 将益蠓误判为毒蠓的代价
    - cost_apf_to_af: 将毒蠓误判为益蠓的代价
    """
    proba_af = 1 - proba_apf

    # 判为Apf的风险
    risk_apf = proba_af * cost_af_to_apf
    # 判为Af的风险
    risk_af = proba_apf * cost_apf_to_af

    return 1 if risk_apf < risk_af else 0, risk_apf, risk_af


print("贝叶斯最小风险决策:")
print(f"  - 误判代价: C(Af→Apf)={1}, C(Apf→Af)={10}")
print(f"  - 决策规则: 选择期望风险较小的类别")
print("-" * 70)
print("样本\tP(Apf)\tR(判为Apf)\tR(判为Af)\t最优决策")
print("-" * 70)

custom_pred = []
for i, proba in enumerate(original_lda_proba[:, 1]):
    pred, risk_apf, risk_af = custom_bayes_classifier(proba)
    custom_pred.append(pred)
    best_label = "Apf(毒蠓)" if pred == 1 else "Af(益蠓)"
    print(f"{i + 1}\t{proba:.4f}\t{risk_apf:.4f}\t\t{risk_af:.4f}\t\t{best_label}")

print(f"\n★ 自定义分类器识别结果: {['Af(益蠓)' if p == 0 else 'Apf(毒蠓)' for p in custom_pred]}")

# ========== 所有方法结果汇总对比 ==========
print("\n" + "=" * 80)
print("问题3 结果汇总对比")
print("=" * 80)

comparison_df = pd.DataFrame({
    '样本': [1, 2, 3],
    '原始LDA(阈值0.5)': ['Apf' if p == 1 else 'Af' for p in original_lda_pred],
    '方法1:阈值0.3': ['Apf' if p == 1 else 'Af' for p in final_pred],
    '方法2:代价矩阵': ['Apf' if p == 1 else 'Af' for p in cost_based_pred],
    '方法3:类别加权': ['Apf' if p == 1 else 'Af' for p in weighted_pred],
    '方法4:贝叶斯风险': ['Apf' if p == 1 else 'Af' for p in custom_pred]
})

print(comparison_df.to_string(index=False))

print("\n" + "=" * 80)
print("结论与建议")
print("=" * 80)
print("""
1. 问题背景: Apf是毒蠓(疾病载体), Af是益蠓(传粉益虫)

2. 误判代价分析:
   - 将毒蠓(Apf)误判为益蠓(Af): 代价极高(疾病传播风险)
   - 将益蠓(Af)误判为毒蠓(Apf): 代价较低(仅浪费资源)

3. 修改后的判别方法:
   ★ 推荐方法: 阈值调整法(阈值=0.3)或代价矩阵法
   ★ 推荐结果: 3个待判样本均判定为 Apf(毒蠓)

4. 防控建议:
   对三个样本所代表的地区，应采取严格防控措施，按毒蠓标准处理。
   宁可错杀益蠓造成资源浪费，也不能漏判毒蠓导致疾病传播。
""")

