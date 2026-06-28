import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False


def draw_dp_table():
    """图1：DP表填充顺序图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 创建5x5网格
    n = 5
    colors = ['#E8F4F8', '#B3D9E8', '#7FB3D5', '#4A8FBB', '#1A6B96']

    for i in range(n):
        for j in range(n):
            if j >= i:
                # 根据区间长度着色
                length = j - i + 1
                color_idx = min(length - 1, 4)
                color = colors[color_idx]
            else:
                color = '#F0F0F0'

            rect = patches.Rectangle((j, n - 1 - i), 1, 1, linewidth=1,
                                     edgecolor='black', facecolor=color)
            ax.add_patch(rect)

            # 填写数字示例
            if i == 0 and j == 0:
                text = "dp[1][1]\n=0"
            elif i == 0 and j == 1:
                text = "dp[1][2]\n=3"
            elif i == 0 and j == 4:
                text = "dp[1][5]\n=33"
            elif j >= i:
                text = f"({i + 1},{j + 1})"
            else:
                text = ""

            ax.text(j + 0.5, n - 1 - i + 0.5, text, ha='center', va='center', fontsize=9)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels([f"j={i + 1}" for i in range(n)])
    ax.set_yticklabels([f"i={i + 1}" for i in range(n - 1, -1, -1)])
    ax.set_title("DP表填充顺序（颜色越深表示区间长度越大）", fontsize=14)

    plt.tight_layout()
    plt.savefig('dp_table.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为: dp_table.png")


def draw_interval_split():
    """图2：区间分割示意图"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    stones = [1, 2, 3, 4, 5]
    n = 5

    # 图2a: k=2的分割
    ax = axes[0]
    for i, val in enumerate(stones):
        rect = patches.Rectangle((i, 0), 0.8, 1, linewidth=1,
                                 edgecolor='black', facecolor='#FFE5B4')
        ax.add_patch(rect)
        ax.text(i + 0.4, 0.5, str(val), ha='center', va='center', fontsize=12)

    # 标注区间
    ax.annotate('', xy=(0, 1.2), xytext=(1.6, 1.2),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax.text(0.8, 1.4, '区间[1,2]', ha='center', color='red', fontsize=10)

    ax.annotate('', xy=(2.4, 1.2), xytext=(4.4, 1.2),
                arrowprops=dict(arrowstyle='<->', color='red'))
    ax.text(3.4, 1.4, '区间[3,5]', ha='center', color='red', fontsize=10)

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('分割方式1: k=2（分割点在第2堆之后）', fontsize=12)

    # 图2b: k=3的分割
    ax = axes[1]
    for i, val in enumerate(stones):
        rect = patches.Rectangle((i, 0), 0.8, 1, linewidth=1,
                                 edgecolor='black', facecolor='#B4E5B4')
        ax.add_patch(rect)
        ax.text(i + 0.4, 0.5, str(val), ha='center', va='center', fontsize=12)

    ax.annotate('', xy=(0, 1.2), xytext=(2.4, 1.2),
                arrowprops=dict(arrowstyle='<->', color='blue'))
    ax.text(1.2, 1.4, '区间[1,3]', ha='center', color='blue', fontsize=10)

    ax.annotate('', xy=(3.2, 1.2), xytext=(4.4, 1.2),
                arrowprops=dict(arrowstyle='<->', color='blue'))
    ax.text(3.8, 1.4, '区间[4,5]', ha='center', color='blue', fontsize=10)

    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('分割方式2: k=3（分割点在第3堆之后）', fontsize=12)

    plt.tight_layout()
    plt.savefig('interval_split.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为: interval_split.png")


def draw_state_transition():
    """图3：状态转移图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义节点位置
    positions = {
        'dp[1][5]': (3, 5),
        'dp[1][1]': (0, 3), 'dp[2][5]': (2, 3),
        'dp[1][2]': (1, 2), 'dp[3][5]': (3, 2),
        'dp[1][3]': (2, 1), 'dp[4][5]': (4, 1),
        'dp[1][4]': (3, 0), 'dp[5][5]': (5, 0),
    }

    # 画节点
    for node, (x, y) in positions.items():
        if node == 'dp[1][5]':
            circle = plt.Circle((x, y), 0.4, color='#FF6B6B', ec='black', lw=2)
        elif '1' in node and '5' in node:
            circle = plt.Circle((x, y), 0.35, color='#FFE5B4', ec='black', lw=1)
        else:
            circle = plt.Circle((x, y), 0.35, color='#B4E5B4', ec='black', lw=1)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=9)

    # 画箭头
    arrows = [
        ('dp[1][5]', 'dp[1][1]'), ('dp[1][5]', 'dp[2][5]'),
        ('dp[1][5]', 'dp[1][2]'), ('dp[1][5]', 'dp[3][5]'),
        ('dp[1][5]', 'dp[1][3]'), ('dp[1][5]', 'dp[4][5]'),
        ('dp[1][5]', 'dp[1][4]'), ('dp[1][5]', 'dp[5][5]'),
    ]

    for start, end in arrows:
        sx, sy = positions[start]
        ex, ey = positions[end]
        ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    # 添加说明文字
    ax.text(3, -0.8, '状态转移方程:\ndp[1][5] = min( dp[1][k] + dp[k+1][5] + sum[1..5] )',
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    ax.set_xlim(-1, 6.5)
    ax.set_ylim(-1.5, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('状态转移图：计算dp[1][5]所需的子问题', fontsize=14)

    plt.tight_layout()
    plt.savefig('state_transition.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为: state_transition.png")


# 运行生成图片
if __name__ == "__main__":
    draw_dp_table()
    draw_interval_split()
    draw_state_transition()
    print("\n所有图片已生成完毕！")