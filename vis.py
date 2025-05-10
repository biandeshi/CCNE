import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patheffects import withStroke

# 原始数据
data = {
    "Method": [
        "DeepSORT", "ByteTrack", "ByteTrack+ReID",
        "UAVMOT", "FairMOT", "TrackFormer", "SORT", "StrongDepthMOT (ours)"
    ],
    "HOTA": [36.92, 40.66, 41.42, 38.13, 31.10, 35.34, 35.08, 44.05],
    "IDF1": [46.71, 50.40, 51.26, 45.15, 37.75, 51.00, 42.84, 56.09],
    "MOTA": [34.40, 39.54, 40.88, 38.69, 12.81, 25.00, 33.15, 39.55]
}

df = pd.DataFrame(data)

# 气泡大小计算
base_sizes = np.interp(df['HOTA'], 
                      (df['HOTA'].min(), df['HOTA'].max()),
                      (800, 2200))
base_sizes[-1] *= 1.5  # 对ours方法气泡放大50%

# 创建图形
plt.figure(figsize=(14, 10))
plt.grid(True, linestyle='--', alpha=0.3)
plt.xlabel('HOTA (%) →', fontsize=16, fontweight='bold')
plt.ylabel('IDF1 (%) ↑', fontsize=16, fontweight='bold')
plt.title('Multiple Object Tracking Performance Comparison', fontsize=20, pad=20)

# 绘制气泡图
scatter = plt.scatter(
    df['HOTA'], df['IDF1'], 
    s=base_sizes,
    c=df['MOTA'],
    cmap='viridis',
    alpha=0.75,
    edgecolors='black',
    linewidth=1.2,
    zorder=3
)

# 标注设置 - 统一连接线长度
uniform_shrinkA = 6  # 统一连接线长度参数
offset_config = [
    {'xytext': (25, 25), 'ha': 'left', 'va': 'bottom', 'shrinkA': uniform_shrinkA, 'shrinkB': 6},  # DeepSORT
    {'xytext': (-30, 25), 'ha': 'right', 'va': 'bottom', 'shrinkA': uniform_shrinkA, 'shrinkB': 6},  # ByteTrack
    {'xytext': (25, -25), 'ha': 'left', 'va': 'top', 'shrinkA': uniform_shrinkA, 'shrinkB': 6},  # ByteTrack+ReID
    {'xytext': (-25, -25), 'ha': 'right', 'va': 'top', 'shrinkA': uniform_shrinkA, 'shrinkB': 6},  # UAVMOT
    {'xytext': (30, 15), 'ha': 'left', 'va': 'bottom', 'shrinkA': uniform_shrinkA, 'shrinkB': 6},  # FairMOT
    {'xytext': (-40, 0), 'ha': 'right', 'va': 'center', 'shrinkA': uniform_shrinkA, 'shrinkB': 6, 'connectionstyle': "arc3,rad=-0.2"},  # TrackFormer
    {'xytext': (-25, 15), 'ha': 'right', 'va': 'bottom', 'shrinkA': uniform_shrinkA, 'shrinkB': 6},  # SORT
    {'xytext': (-35, -15), 'ha': 'right', 'va': 'top', 'shrinkA': 0, 'shrinkB': 6}  # StrongDepthMOT (保持从边缘伸出)
]

text_effect = withStroke(linewidth=3, foreground='white')

for i, row in df.iterrows():
    config = offset_config[i]
    is_ours = "(ours)" in row['Method']
    
    plt.annotate(
        f"{row['Method']}\nHOTA: {row['HOTA']:.1f}\nIDF1: {row['IDF1']:.1f}",
        (row['HOTA'], row['IDF1']),
        xytext=config['xytext'],
        textcoords='offset points',
        fontsize=14,
        ha=config['ha'],
        va=config['va'],
        bbox=dict(
            boxstyle='round,pad=0.5',
            fc='gold' if is_ours else 'white',
            alpha=0.95,
            ec='red' if is_ours else 'gray',
            lw=1.0 if is_ours else 0.7
        ),
        arrowprops=dict(
            arrowstyle='-',
            color='red' if is_ours else 'gray',
            lw=1.2 if is_ours else 1.0,
            shrinkA=config['shrinkA'],
            shrinkB=config['shrinkB'],
            connectionstyle=config.get('connectionstyle', None)
        ),
        path_effects=[text_effect],
        linespacing=1.2
    )

# 颜色条设置
cbar = plt.colorbar(scatter, shrink=0.6, pad=0.02)
cbar.set_label('MOTA (%)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# 创建图例元素
legend_elements = [
    plt.scatter([], [], s=600, c='gray', alpha=0.7, label=f"Min HOTA: {df['HOTA'].min():.1f}%"),
    plt.scatter([], [], s=1200, c='gray', alpha=0.7, label=f"Max HOTA: {df['HOTA'].max():.1f}%")
]

# 调整图例布局 - 固定在右下角
legend = plt.legend(
    handles=legend_elements,
    title="Bubble Size Indicator",
    fontsize=12,
    title_fontsize=13,
    loc='lower right',
    framealpha=1,
    bbox_to_anchor=(0.98, 0.02),  # 调整到右下角位置
    handletextpad=3.5,
    borderpad=2.0,
    labelspacing=1.8,
    handlelength=0,
    markerscale=1.0,
    ncol=1
)

# 设置不同的气泡大小
for handle, size in zip(legend.legend_handles, [600, 1200]):
    handle.set_sizes([size])
    handle.set_edgecolor('black')
    handle.set_linewidth(0.8)

# 保持原始坐标轴范围
plt.xlim(30, 46)
plt.ylim(37, 58)

plt.tight_layout()
plt.savefig('MOT_comparison_final.pdf', 
            format='pdf', 
            bbox_inches='tight',
            dpi=600)
plt.show()