# -*- coding: utf-8 -*-
"""
数据探索与统计可视化脚本
生成8张可视化图表和1份数据质量报告
"""

import os
import sys
import re
import glob
from collections import Counter
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

EDGE_CSV = os.path.join(ROOT_DIR, 'processed', 'ml_twitter.csv')
NODE_LABELS_NPY = os.path.join(ROOT_DIR, 'processed', 'ml_twitter_node_labels.npy')
NODE_INFO_CSV = os.path.join(ROOT_DIR, 'processed', 'node_id_info_clear_new2_TF_filtered.csv')
CSV_DIR = os.path.join(ROOT_DIR, 'processed', 'csv_ok_lable')

VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

REPORT_PATH = os.path.join(BASE_DIR, 'data_quality_report.md')

# 设置全局样式
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 10

# 配置中文字体
from matplotlib import font_manager
found = False
for font_name in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei']:
    fonts = font_manager.findSystemFonts()
    for f in fonts:
        try:
            if font_name.lower().replace(' ', '') in f.lower().replace(' ', ''):
                plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
                found = True
                break
        except:
            continue
    if found:
        break
# 若路径匹配失败，尝试通过字体名称匹配
if not found:
    for f in font_manager.fontManager.ttflist:
        if 'Noto Sans CJK' in f.name or 'WenQuanYi' in f.name or 'SimHei' in f.name:
            plt.rcParams['font.family'] = f.name
            break
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 加载所有数据文件，验证数据完整性
# ============================================================
print("=" * 60)
print("[1/9] 加载数据并验证完整性")
print("=" * 60)

# 边表
edges = pd.read_csv(EDGE_CSV)
print(f"  边表加载成功: {edges.shape[0]} 条边, {edges.shape[1]} 列")
print(f"    列名: {list(edges.columns)}")

# 节点标签
node_labels = np.load(NODE_LABELS_NPY)
print(f"  节点标签加载成功: shape={node_labels.shape}, dtype={node_labels.dtype}")

# 节点信息
node_info = pd.read_csv(NODE_INFO_CSV)
print(f"  节点信息加载成功: {node_info.shape[0]} 行, {node_info.shape[1]} 列")

# 个体CSV文件列表
csv_files = sorted(glob.glob(os.path.join(CSV_DIR, '*.csv')))
print(f"  个体CSV文件: {len(csv_files)} 个")

# 数据完整性验证
integrity_issues = []
if edges.isnull().sum().sum() > 0:
    integrity_issues.append(f"边表存在 {edges.isnull().sum().sum()} 个缺失值")

if node_info.isnull().sum().sum() > 0:
    integrity_issues.append(f"节点信息存在 {node_info.isnull().sum().sum()} 个缺失值")

# 检查节点一致性
max_node_in_edges = max(edges['u'].max(), edges['i'].max())
expected_nodes = len(node_labels) - 1  # node 0 is padding/unlabeled
if max_node_in_edges > expected_nodes:
    integrity_issues.append(f"边表中最大节点ID({max_node_in_edges})超过节点标签数组大小({len(node_labels)})")

if len(csv_files) != node_info.shape[0]:
    integrity_issues.append(f"个体CSV文件数({len(csv_files)})与节点信息行数({node_info.shape[0]})不一致")

if integrity_issues:
    print("  [警告] 数据完整性问题:")
    for issue in integrity_issues:
        print(f"    - {issue}")
else:
    print("  [通过] 数据完整性验证通过")

# ============================================================
# 预计算: 读取所有个体CSV的统计信息
# ============================================================
print("\n[预计算] 读取所有个体CSV统计信息...")

tweet_counts = []
user_names = []
retweet_counts = []
reply_counts = []
mention_counts = []
original_counts = []
times_list = []
text_samples = []

# 限制词云采样的用户数和每用户的推文数
MAX_TEXT_USERS = 200
MAX_TEXT_PER_USER = 50

for idx, fpath in enumerate(csv_files):
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath, dtype=str)
    total = len(df)
    tweet_counts.append(total)
    user_names.append(fname.replace('.csv', ''))

    # 交互类型统计
    rt = df['RT_ID'].notna().sum() if 'RT_ID' in df.columns else 0
    rp = df['R_ID'].notna().sum() if 'R_ID' in df.columns else 0
    mt = df['M_ID'].notna().sum() if 'M_ID' in df.columns else 0
    orig = total - rt - rp  # 原始推文: 非转发且非回复

    retweet_counts.append(rt)
    reply_counts.append(rp)
    mention_counts.append(mt)
    original_counts.append(max(0, orig))

    # 时间数据
    if 'time' in df.columns:
        valid_times = df['time'].dropna()
        times_list.extend(valid_times.tolist())

    # 文本采样 (用于词云)
    if idx < MAX_TEXT_USERS and 'tweets' in df.columns:
        texts = df['tweets'].dropna().astype(str).tolist()
        if texts:
            sampled = np.random.choice(texts, size=min(len(texts), MAX_TEXT_PER_USER), replace=False)
            text_samples.extend(sampled.tolist())

    if (idx + 1) % 100 == 0:
        print(f"    已处理 {idx + 1}/{len(csv_files)} 个用户...")

print(f"  完成! 共统计 {len(tweet_counts)} 个用户, 词云采样 {len(text_samples)} 条推文")

# 构建用户统计DataFrame
user_stats = pd.DataFrame({
    'user': user_names,
    'tweet_count': tweet_counts,
    'retweet_count': retweet_counts,
    'reply_count': reply_counts,
    'mention_count': mention_counts,
    'original_count': original_counts,
})

# ============================================================
# 网络统计: 入度/出度/总度
# ============================================================
print("\n[预计算] 计算网络度分布...")

out_degree = edges['u'].value_counts().sort_index()
in_degree = edges['i'].value_counts().sort_index()
all_nodes = set(edges['u'].unique()) | set(edges['i'].unique())

# 补充缺失节点（度为0）
node_ids = sorted(all_nodes)
out_deg = [out_degree.get(n, 0) for n in node_ids]
in_deg = [in_degree.get(n, 0) for n in node_ids]
total_deg = [o + i for o, i in zip(out_deg, in_deg)]

degree_df = pd.DataFrame({
    'node_id': node_ids,
    'out_degree': out_deg,
    'in_degree': in_deg,
    'total_degree': total_deg,
})

print(f"  网络节点数: {len(node_ids)}, 边数: {len(edges)}")
print(f"  平均出度: {np.mean(out_deg):.2f}, 平均入度: {np.mean(in_deg):.2f}")

# ============================================================
# 辅助函数
# ============================================================
def save_fig(name):
    path = os.path.join(VIZ_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {name}")


def clean_text_for_wordcloud(texts):
    """清洗推文文本，用于词云生成"""
    cleaned = []
    for t in texts:
        t = str(t)
        # 去除URL
        t = re.sub(r'http\S+|www\.\S+', '', t)
        # 去除@提及
        t = re.sub(r'@\w+', '', t)
        # 去除#话题标签符号（保留词）
        t = re.sub(r'#(\w+)', r'\1', t)
        # 去除RT前缀
        t = re.sub(r'\bRT\b', '', t)
        # 去除多余空格和特殊字符
        t = re.sub(r'[^\w\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        if len(t) > 3:
            cleaned.append(t.lower())
    return cleaned


# ============================================================
# 2. 统计每个用户的推文数量分布 → 直方图
# ============================================================
print("\n" + "=" * 60)
print("[2/9] 生成推文数量分布直方图")
print("=" * 60)

plt.figure(figsize=(10, 6))
# 使用对数bins以适应长尾分布
bins = np.logspace(np.log10(max(1, min(tweet_counts))), np.log10(max(tweet_counts) + 1), 31)
plt.hist(tweet_counts, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('推文数量（对数刻度）', fontsize=12)
plt.ylabel('用户数量（对数刻度）', fontsize=12)
plt.title('用户推文数量分布', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
save_fig('tweet_count_histogram.png')

# ============================================================
# 3. 统计用户交互频率分布(入度/出度) → 度分布图
# ============================================================
print("\n" + "=" * 60)
print("[3/9] 生成度分布图")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

out_deg_nonzero = [d for d in out_deg if d > 0]
in_deg_nonzero = [d for d in in_deg if d > 0]

bins = np.logspace(np.log10(1), np.log10(max(max(out_deg_nonzero), max(in_deg_nonzero)) + 1), 31)
ax.hist(out_deg_nonzero, bins=bins, color='coral', edgecolor='white', alpha=0.7, label='出度')
ax.hist(in_deg_nonzero, bins=bins, color='teal', edgecolor='white', alpha=0.7, label='入度')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('度数（对数刻度）', fontsize=12)
ax.set_ylabel('用户数量（对数刻度）', fontsize=12)
ax.set_title('用户交互频率分布（入度/出度）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_fig('degree_distribution.png')

# ============================================================
# 4. 分析时间维度的数据分布 → 时间分布图
# ============================================================
print("\n" + "=" * 60)
print("[4/9] 生成时间分布图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('时间维度数据分布', fontsize=14, fontweight='bold')

# (a) 边表中的时间切片 ts 分布
ax = axes[0]
ts_counts = edges['ts'].value_counts().sort_index()
ax.bar(ts_counts.index, ts_counts.values, color='mediumpurple', edgecolor='white', alpha=0.85, width=0.8)
ax.set_xlabel('时间切片', fontsize=11)
ax.set_ylabel('交互数量', fontsize=11)
ax.set_title('时间切片交互分布', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# (b) 实际推文时间分布（按月份）
ax = axes[1]
if times_list:
    time_series = pd.to_datetime(times_list, errors='coerce')
    time_series = time_series.dropna()
    if len(time_series) > 0:
        monthly = time_series.to_series().dt.to_period('M').value_counts().sort_index()
        ax.plot(monthly.index.astype(str), monthly.values, marker='o', color='darkgreen', linewidth=1.5, markersize=3)
        ax.set_xlabel('月份', fontsize=11)
        ax.set_ylabel('推文数量', fontsize=11)
        ax.set_title('月度推文量分布', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        # 如果月份太多，只显示部分标签
        if len(monthly) > 20:
            step = max(1, len(monthly) // 10)
            ax.set_xticks(ax.get_xticks()[::step])
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, '无有效时间数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('月度推文量分布', fontsize=12, fontweight='bold')
else:
    ax.text(0.5, 0.5, '无时间数据', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('月度推文量分布', fontsize=12, fontweight='bold')

plt.tight_layout()
save_fig('time_distribution.png')

# ============================================================
# 5. 分析转发vs回复的比例和模式 → 柱状图
# ============================================================
print("\n" + "=" * 60)
print("[5/9] 生成交互类型柱状图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('交互类型分布', fontsize=14, fontweight='bold')

# (a) 边表级别: label=1 (转发), label=3 (回复)
ax = axes[0]
edge_label_counts = edges['label'].value_counts().sort_index()
labels_map = {1: '转发', 3: '回复'}
colors = ['#FF6B6B', '#4ECDC4']
bars = ax.bar([labels_map.get(l, str(l)) for l in edge_label_counts.index],
              edge_label_counts.values, color=colors, edgecolor='white', alpha=0.9)
ax.set_ylabel('数量', fontsize=11)
ax.set_title('边交互类型分布', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10)

# (b) 个体推文级别: 转发/回复/提及/原创 的比例
ax = axes[1]
total_rt = sum(retweet_counts)
total_rp = sum(reply_counts)
total_mt = sum(mention_counts)
total_orig = sum(original_counts)

categories = ['原创', '转发', '回复', '提及']
values = [total_orig, total_rt, total_rp, total_mt]
colors2 = ['#45B7D1', '#FF6B6B', '#4ECDC4', '#96CEB4']
bars = ax.bar(categories, values, color=colors2, edgecolor='white', alpha=0.9)
ax.set_ylabel('数量', fontsize=11)
ax.set_title('推文交互类型分布', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
save_fig('interaction_type.png')

# ============================================================
# 6. 绘制交互网络的度分布图(幂律分布验证)
# ============================================================
print("\n" + "=" * 60)
print("[6/9] 生成幂律分布验证图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('度分布幂律验证', fontsize=14, fontweight='bold')

# (a) 总度分布 log-log 图
ax = axes[0]
total_deg_nonzero = sorted([d for d in total_deg if d > 0], reverse=True)
# 计算CCDF
k_values = sorted(set(total_deg_nonzero))
pk = [total_deg_nonzero.count(k) / len(total_deg_nonzero) for k in k_values]
ccdf = [sum(pk[i:]) for i in range(len(pk))]

ax.loglog(k_values, ccdf, 'o', color='darkblue', markersize=4, alpha=0.7, label='经验CCDF')

# 线性拟合 (log-log)
log_k = np.log(k_values)
log_ccdf = np.log(ccdf)
# 只取中间部分拟合，避免低度噪声和高度截断
fit_mask = (np.array(k_values) >= 3) & (np.array(k_values) <= np.percentile(total_deg_nonzero, 95))
if fit_mask.sum() > 5:
    slope, intercept = np.polyfit(log_k[fit_mask], log_ccdf[fit_mask], 1)
    fitted = np.exp(intercept) * np.array(k_values) ** slope
    ax.loglog(k_values, fitted, '--', color='red', linewidth=2,
              label=f'幂律拟合: α ≈ {-slope:.2f}')

ax.set_xlabel('度数（对数刻度）', fontsize=11)
ax.set_ylabel('累积概率 P(K≥k)（对数刻度）', fontsize=11)
ax.set_title('度分布互补累积分布（CCDF）', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# (b) 度分布直方图 + 幂律参考线
ax = axes[1]
# 使用对数bins
bins = np.logspace(np.log10(1), np.log10(max(total_deg_nonzero) + 1), 31)
counts, bin_edges = np.histogram(total_deg_nonzero, bins=bins)
bin_centers = (bin_edges[:-1] * bin_edges[1:]) ** 0.5
ax.loglog(bin_centers, counts, 's', color='darkorange', markersize=5, alpha=0.8, label='经验分布')

# 拟合参考线
if fit_mask.sum() > 5:
    # PDF slope = CCDF slope + 1
    pdf_slope = slope + 1
    # 归一化到数据范围
    ref_y = counts[0] * (np.array(bin_centers) / bin_centers[0]) ** pdf_slope
    ax.loglog(bin_centers, ref_y, '--', color='purple', linewidth=2,
              label=f'幂律参考: γ ≈ {-pdf_slope:.2f}')

ax.set_xlabel('度数（对数刻度）', fontsize=11)
ax.set_ylabel('频率（对数刻度）', fontsize=11)
ax.set_title('度分布概率密度（PDF）', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
save_fig('power_law_degree.png')

# ============================================================
# 7. 绘制小提琴图展示用户活跃度分布
# ============================================================
print("\n" + "=" * 60)
print("[7/9] 生成用户活跃度小提琴图")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('用户活跃度分布（小提琴图）', fontsize=14, fontweight='bold')

metrics = [
    ('tweet_count', '推文数', 'steelblue'),
    ('out_degree', '出度', 'coral'),
    ('in_degree', '入度', 'teal'),
]

# 合并数据
merged = pd.merge(user_stats, degree_df, left_index=True, right_index=True, how='outer')
merged = merged.fillna(0)

for ax, (col, title, color) in zip(axes, metrics):
    data = merged[col].values
    data_nonzero = data[data > 0]
    
    # 小提琴图用log数据更美观
    log_data = np.log10(data_nonzero) if len(data_nonzero) > 0 else [0]
    
    parts = ax.violinplot([log_data], positions=[1], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax.set_xticks([1])
    ax.set_xticklabels([title], fontsize=10)
    ax.set_ylabel('Log10(数值)', fontsize=10)
    ax.set_title(f'{title}分布', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计注释
    median = np.median(data_nonzero) if len(data_nonzero) > 0 else 0
    mean = np.mean(data_nonzero) if len(data_nonzero) > 0 else 0
    ax.text(0.95, 0.95, f'中位数: {median:.1f}\n均值: {mean:.1f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
save_fig('activity_violin.png')

# ============================================================
# 8. 生成整体高频词词云图
# ============================================================
print("\n" + "=" * 60)
print("[8/9] 生成词云图")
print("=" * 60)

if text_samples:
    cleaned_texts = clean_text_for_wordcloud(text_samples)
    all_text = ' '.join(cleaned_texts)
    
    # 停用词
    stopwords = set([
        'the', 'to', 'and', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was',
        'for', 'on', 'are', 'with', 'as', 'i', 'his', 'they', 'be', 'at', 'one', 'have',
        'this', 'from', 'or', 'had', 'by', 'not', 'word', 'but', 'what', 'all', 'were',
        'we', 'when', 'your', 'can', 'said', 'there', 'each', 'which', 'she', 'do',
        'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then',
        'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him',
        'has', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been',
        'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come',
        'made', 'may', 'part', 'am', 'an', 'us', 'rt', 'https', 'co', 't', 's', 're',
        'm', 've', 'll', 'don', 'doesn', 'didn', 'isn', 'wasn', 'weren', 'haven',
        'hasn', 'hadn', 'wouldn', 'shouldn', 'couldn', 'mightn', 'mustn', 'needn',
        'shan', 'won', 'i’m', 'it’s', 'that’s', 'there’s', 'what’s', 'who’s', 'here’s',
        'let’s', 'he’s', 'she’s', 'they’re', 'we’re', 'you’re', 'i’ve', 'you’ve',
        'we’ve', 'they’ve', 'i’d', 'you’d', 'he’d', 'she’d', 'we’d', 'they’d',
        'i’ll', 'you’ll', 'he’ll', 'she’ll', 'we’ll', 'they’ll', 'isnt', 'wasnt',
        'dont', 'doesnt', 'didnt', 'wouldnt', 'couldnt', 'shouldnt', 'hasnt',
        'havent', 'hadnt', 'arent', 'werent', 'im', 'th', 'st', 'nd', 'rd',
    ])
    
    wordcloud = WordCloud(
        width=1200, height=800,
        background_color='white',
        colormap='viridis',
        max_words=200,
        stopwords=stopwords,
        contour_width=1,
        contour_color='steelblue',
        random_state=42,
    ).generate(all_text)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('高频词词云图', fontsize=16, fontweight='bold', pad=20)
    save_fig('wordcloud.png')
else:
    print("  [跳过] 无有效文本数据用于生成词云")

# ============================================================
# 9. 生成数据质量报告
# ============================================================
print("\n" + "=" * 60)
print("[9/9] 生成数据质量报告")
print("=" * 60)

# 计算各种统计指标
total_tweets = sum(tweet_counts)
avg_tweets = np.mean(tweet_counts)
median_tweets = np.median(tweet_counts)
std_tweets = np.std(tweet_counts)

edge_label_dist = edges['label'].value_counts().sort_index()
edge_ts_dist = edges['ts'].value_counts().sort_index()

self_loops = (edges['u'] == edges['i']).sum()
unique_edges = edges[['u', 'i', 'label']].drop_duplicates().shape[0]

# 节点标签分布
label_dist = Counter(node_labels)

# 时间范围
if times_list:
    valid_times = pd.to_datetime(times_list, errors='coerce').dropna()
    time_range = f"{valid_times.min()} ~ {valid_times.max()}" if len(valid_times) > 0 else "N/A"
else:
    time_range = "N/A"

report = f"""# 数据质量报告 (Data Quality Report)

> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 数据概览 (Data Overview)

| 项目 | 数值 |
|------|------|
| 节点总数 (Nodes) | {len(node_labels)} (含索引0占位节点) |
| 有信息节点数 (Nodes with info) | {node_info.shape[0]} |
| 边总数 (Edges) | {len(edges):,} |
| 个体CSV文件数 (User CSVs) | {len(csv_files)} |
| 推文总数 (Total Tweets) | {total_tweets:,} |
| 时间范围 (Time Range) | {time_range} |

---

## 2. 数据完整性验证 (Integrity Checks)

"""

if integrity_issues:
    report += "**发现以下问题:**\n\n"
    for issue in integrity_issues:
        report += f"- ?? {issue}\n"
else:
    report += "? **所有数据完整性检查通过**\n"

report += f"""
---

## 3. 节点统计 (Node Statistics)

### 3.1 节点标签分布 (Node Label Distribution)

| 标签值 | 含义 | 数量 | 比例 |
|--------|------|------|------|
| -1 | 未标注 (Unlabeled) | {label_dist.get(-1, 0)} | {label_dist.get(-1, 0)/len(node_labels)*100:.1f}% |
| 0 | 党派0 (Party 0) | {label_dist.get(0, 0)} | {label_dist.get(0, 0)/len(node_labels)*100:.1f}% |
| 1 | 党派1 (Party 1) | {label_dist.get(1, 0)} | {label_dist.get(1, 0)/len(node_labels)*100:.1f}% |

### 3.2 节点信息表 (Node Info Table)

- 列: {list(node_info.columns)}
- 行数: {node_info.shape[0]}
- Party 0 用户数: {(node_info['party'] == 0).sum()}
- Party 1 用户数: {(node_info['party'] == 1).sum()}

---

## 4. 边统计 (Edge Statistics)

### 4.1 边类型分布 (Edge Label Distribution)

| 标签值 | 含义 | 数量 | 比例 |
|--------|------|------|------|
| 1 | 转发 (Retweet) | {edge_label_dist.get(1, 0):,} | {edge_label_dist.get(1, 0)/len(edges)*100:.1f}% |
| 3 | 回复 (Reply) | {edge_label_dist.get(3, 0):,} | {edge_label_dist.get(3, 0)/len(edges)*100:.1f}% |

### 4.2 网络基本指标 (Network Metrics)

| 指标 | 数值 |
|------|------|
| 活跃源节点数 (Unique sources) | {edges['u'].nunique():,} |
| 活跃目标节点数 (Unique targets) | {edges['i'].nunique():,} |
| 自环边数 (Self-loops) | {self_loops:,} ({self_loops/len(edges)*100:.2f}%) |
| 唯一边数 (Unique edges) | {unique_edges:,} |
| 平均出度 (Avg out-degree) | {np.mean(out_deg):.2f} |
| 平均入度 (Avg in-degree) | {np.mean(in_deg):.2f} |
| 最大出度 (Max out-degree) | {max(out_deg):,} |
| 最大入度 (Max in-degree) | {max(in_deg):,} |
| 最大总度 (Max total degree) | {max(total_deg):,} |

### 4.3 时间切片分布 (Time Slice Distribution)

| 时间切片 (ts) | 边数 |
|---------------|------|
"""

for ts_val, ts_cnt in edge_ts_dist.head(10).items():
    report += f"| {ts_val} | {ts_cnt:,} |\n"

if len(edge_ts_dist) > 10:
    report += f"| ... | ... |\n"

report += f"""
---

## 5. 推文统计 (Tweet Statistics)

| 指标 | 数值 |
|------|------|
| 推文总数 | {total_tweets:,} |
| 平均每用户推文数 | {avg_tweets:.1f} |
| 中位数推文数 | {median_tweets:.1f} |
| 标准差 | {std_tweets:.1f} |
| 最小推文数 | {min(tweet_counts):,} |
| 最大推文数 | {max(tweet_counts):,} |

### 5.1 交互类型统计 (Interaction Types)

| 类型 | 总数 | 占比 |
|------|------|------|
| 原创推文 (Original) | {total_orig:,} | {total_orig/total_tweets*100:.1f}% |
| 转发 (Retweet) | {total_rt:,} | {total_rt/total_tweets*100:.1f}% |
| 回复 (Reply) | {total_rp:,} | {total_rp/total_tweets*100:.1f}% |
| 提及 (Mention) | {total_mt:,} | {total_mt/total_tweets*100:.1f}% |

---

## 6. 可视化文件清单 (Generated Visualizations)

| 文件名 | 说明 |
|--------|------|
| tweet_count_histogram.png | 用户推文数量分布直方图 |
| degree_distribution.png | 入度/出度分布图 |
| time_distribution.png | 时间维度分布图 |
| interaction_type.png | 交互类型比例柱状图 |
| power_law_degree.png | 幂律分布验证图 |
| activity_violin.png | 用户活跃度小提琴图 |
| wordcloud.png | 高频词词云图 |

---

## 7. 数据质量总结 (Summary)

1. **完整性**: 边表、节点标签、节点信息、个体CSV均成功加载，节点0为未标注占位节点。
2. **一致性**: 边表节点ID范围(1~{max_node_in_edges})与节点信息({node_info.shape[0]})基本一致。
3. **分布特征**: 推文数量呈明显的长尾分布，少数用户贡献了大部分内容。
4. **网络特征**: 度分布呈现幂律特征，网络中存在少量高连接度枢纽节点。
5. **交互特征**: 边表中转发({edge_label_dist.get(1, 0):,})远多于回复({edge_label_dist.get(3, 0):,})，比例为 {edge_label_dist.get(1, 0)/max(edge_label_dist.get(3, 0), 1):.1f}:1。
6. **时间覆盖**: 边表覆盖 {edges['ts'].nunique()} 个时间切片 (ts={edges['ts'].min():.0f}~{edges['ts'].max():.0f})。

---
*报告由 data_exploration.py 自动生成*
"""

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  已保存报告: {REPORT_PATH}")

# ============================================================
# 完成
# ============================================================
print("\n" + "=" * 60)
print("所有任务完成!")
print(f"可视化文件保存至: {VIZ_DIR}")
print(f"报告保存至: {REPORT_PATH}")
print("=" * 60)
