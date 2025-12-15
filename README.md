# GB_System
# _*_coding:utf-8 _*_
import warnings
from sklearn.cluster import k_means
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# === 忽略警告 ===
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==========================================
#   系统基础算子
# ==========================================

def distances(data, p):
    return ((data - p) ** 2).sum(axis=0) ** 0.5

#定义系统熵的计算算子 (测量工具)
def calculate_entropy(data):
    """
    [系统工程-热力学] 计算系统的熵值
    """
    n = data.shape[0]
    if n == 0: return 0.0

    labels = data[:, 0]
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / n
    entropy = -np.sum(probs * np.log2(probs + 1e-9))
    return entropy


def get_label_and_purity(data):
    num = data.shape[0]
    if num == 0: return 0, 0
    num_positive = np.sum(data[:, 0] == 1)
    num_negative = np.sum(data[:, 0] == 0)
    purity = max(num_positive, num_negative) / num
    label = 1 if num_positive >= num_negative else 0
    return label, purity


def calculate_center_and_radius(granular_ball):
    data_no_label = granular_ball[:, 1:]
    if data_no_label.shape[0] == 0:
        return np.zeros(data_no_label.shape[1]), 0.0
    center = data_no_label.mean(0)
    dists = np.sqrt(np.sum((data_no_label - center) ** 2, axis=1))
    radius = np.max(dists) if len(dists) > 0 else 0.0
    return center, radius


def plot_gb(granular_ball_list, dataset_name=""):
    color = {0: 'r', 1: 'k'}
    plt.figure(figsize=(10, 8))
    # plt.axis('equal') # 如果数据分布很扁，可以注释掉这行

    for granular_ball in granular_ball_list:
        if granular_ball.shape[0] == 0: continue
        label, p = get_label_and_purity(granular_ball)
        center, radius = calculate_center_and_radius(granular_ball)

        # 绘制数据点 (只画前两个特征维度，方便可视化)
        data0 = granular_ball[granular_ball[:, 0] == 0]
        data1 = granular_ball[granular_ball[:, 0] == 1]

        # 检查维度，防止报错
        dim_x = 1  # 特征1的索引
        dim_y = 2  # 特征2的索引

        if granular_ball.shape[1] > 2:
            if len(data0) > 0: plt.plot(data0[:, dim_x], data0[:, dim_y], '.', color=color[0], markersize=5)
            if len(data1) > 0: plt.plot(data1[:, dim_x], data1[:, dim_y], '.', color=color[1], markersize=5)

            # 绘制粒球边界 (圆)
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[dim_x - 1] + radius * np.cos(theta)
            y = center[dim_y - 1] + radius * np.sin(theta)
            plt.plot(x, y, color[label], linewidth=0.8)

            # 绘制球心
            plt.plot(center[dim_x - 1], center[dim_y - 1], 'x', color=color[label])

    plt.title(f"Granular Ball System Visualization - {dataset_name}")
    print(f"  [提示] 请手动关闭弹出的图片窗口，以便程序继续运行下一个数据集...")
    plt.show()


# ==========================================
#   系统演化核心
# ==========================================

def split_ball(data, min_samples=3):
    if data.shape[0] < min_samples:
        return [data]

    data_no_label = data[:, 1:]
    target_k = 2

    try:
        kmeans = k_means(X=data_no_label, n_clusters=target_k, random_state=None, n_init=1)
        label_cluster = kmeans[1]
    except:
        return [data]

    new_balls = []
    actual_labels = np.unique(label_cluster)

    if len(actual_labels) < 2:
        return [data]

    for i in actual_labels:
        sub_ball = data[label_cluster == i, :]
        if sub_ball.shape[0] > 0:
            new_balls.append(sub_ball)

    return new_balls


def predict_accuracy(granular_ball_list, test_data):
    if len(granular_ball_list) == 0: return 0.0

    test_X = test_data[:, 1:]
    test_y = test_data[:, 0]

    centers = []
    labels = []

    for ball in granular_ball_list:
        if ball.shape[0] == 0: continue
        centers.append(ball[:, 1:].mean(0))
        l, _ = get_label_and_purity(ball)
        labels.append(l)

    centers = np.array(centers)
    labels = np.array(labels)

    if len(centers) == 0: return 0.0

    test_X_exp = test_X[:, np.newaxis, :]
    centers_exp = centers[np.newaxis, :, :]

    dists = np.sum((test_X_exp - centers_exp) ** 2, axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    preds = labels[nearest_idx]

    return np.mean(preds == test_y)


# ==========================================
#   主控制程序
# ==========================================

def main():
    try:
        data_mat = scipy.io.loadmat(r'dataset16.mat')
    except FileNotFoundError:
        print("未找到 dataset16.mat")
        return

    # 这里更新为你需要的 7 个数据集
    keys = ['fourclass', 'mushrooms', 'breastcancer', 'codrna', 'svmguide1', 'votes', 'svmguide3']

    for k in keys:
        if k not in data_mat:
            print(f"提示: 数据集 {k} 不在文件中，跳过。")
            continue

        print(f"\n========== 正在处理复杂系统: {k} ==========")

        # 1. 系统输入处理
        raw_data = data_mat[k]
        if hasattr(raw_data, 'toarray'): raw_data = raw_data.toarray()
        raw_data = raw_data.astype(float)
        raw_data[raw_data[:, 0] == -1, 0] = 0

        y = raw_data[:, 0]
        X = raw_data[:, 1:]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        data_processed = np.column_stack((y, X_scaled))

        # 2. 数据集划分
        train_val, test_data = train_test_split(data_processed, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(train_val, test_size=0.25, random_state=42)

        start = time.time()

        # === 初始化状态 ===
        purity_threshold = 0.90
        entropy_threshold = 0.1
        #这是系统的**“温度计”。它利用香农熵（Shannon Entropy）公式来量化当前粒球内部的混乱程度（Disorder）**

        granular_ball_list = [train_data]
        best_val_acc = 0.0
        patience = 2
        patience_counter = 0
        iteration = 0

        print(f"  [启动闭环控制] 初始球数: 1, 纯度阈值: {purity_threshold}")

        while True:
            iteration += 1
            new_list = []
            split_happened = False

            for ball in granular_ball_list:
                _, p = get_label_and_purity(ball)
                h = calculate_entropy(ball)

                if p < purity_threshold or h > entropy_threshold:
                    split_res = split_ball(ball, min_samples=5)
                    new_list.extend(split_res)
                    if len(split_res) > 1:
                        split_happened = True
                else:
                    new_list.append(ball)

            granular_ball_list = new_list
            current_val_acc = predict_accuracy(granular_ball_list, val_data)

            if not split_happened:
                print("  [系统稳态] 粒球不再分裂，演化结束。")
                break

            if current_val_acc > best_val_acc + 0.001:
                best_val_acc = current_val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [负反馈触发] 检测到过拟合风险，强制终止演化。")
                    break

        end = time.time()

        final_test_acc = predict_accuracy(granular_ball_list, test_data)

        print(f"  [最终状态] 粒球数: {len(granular_ball_list)}")
        print(f"  [性能指标] 验证集精度: {best_val_acc * 100:.2f}%, 测试集精度: {final_test_acc * 100:.2f}%")
        print(f"  [系统耗时] {(end - start) * 1000:.1f} ms")

        # === 修改点：移除 if k == keys[0] 限制，所有数据集都绘图 ===
        print(f"  [绘图] 正在生成 {k} 的粒球分布图...")
        try:
            plot_gb(granular_ball_list, dataset_name=k)
        except Exception as e:
            print(f"  [绘图失败] 数据集 {k} 无法绘图 (可能特征维度不足或数据为空): {e}")

        print("-" * 40)

if __name__ == '__main__':
    main()
