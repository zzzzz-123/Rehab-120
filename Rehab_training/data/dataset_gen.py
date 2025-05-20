import os
import numpy as np

data_folder = r'D:\pycharm project\Rehab-120\Rehab_training\data'  # 包含16个滑窗后的 .npy 文件的目录

X_list = []
y_list = []

file_list = sorted(os.listdir(data_folder))  # 确保类别顺序一致（0~15）
for class_idx, file in enumerate(file_list):
    if not file.endswith('.npy'):
        continue

    file_path = os.path.join(data_folder, file)
    data = np.load(file_path)  # shape: (N_i, 200, 6)

    X_list.append(data)
    y_list.append(np.full((data.shape[0],), class_idx))  # 为该类生成标签

# 合并所有类别的数据
X = np.concatenate(X_list, axis=0)  # shape: (total_samples, 200, 6)
y = np.concatenate(y_list, axis=0)  # shape: (total_samples,)

# 打印确认
print(f"X shape: {X.shape}, y shape: {y.shape}, number of classes: {len(np.unique(y))}")

# 保存
np.save(os.path.join(data_folder, 'X.npy'), X)
np.save(os.path.join(data_folder, 'y.npy'), y)
