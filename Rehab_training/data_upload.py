import os
import numpy as np

# 设置原始数据目录和输出目录
data_dir = r"C:\Users\Administrator\Desktop\Rehab_exercise\d02_processed_data"
save_dir = r"C:\Users\Administrator\Desktop\Rehab_exercise\d02_processed_data"
os.makedirs(save_dir, exist_ok=True)

# 获取文件列表并排序，确保顺序匹配
file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])

# 检查是否能分成 16 组（每组两个文件）
assert len(file_list) == 32, f"文件数量应为 32 个，现在是 {len(file_list)} 个。"
assert len(file_list) % 2 == 0, "文件数不是偶数，无法两两配对。"

# 逐组处理
for i in range(0, len(file_list), 2):
    file1 = os.path.join(data_dir, file_list[i])
    file2 = os.path.join(data_dir, file_list[i + 1])

    data1 = np.load(file1)  # shape: (N, 880, 6)
    data2 = np.load(file2)  # shape: (N, 880, 6)

    # 检查样本数一致
    assert data1.shape[0] == data2.shape[0], f"{file_list[i]} 与 {file_list[i+1]} 样本数不一致"

    # 通道拼接 => (N, 880, 12)
    combined = np.concatenate((data1, data2), axis=2)

    # 保存为单个动作文件
    action_idx = i // 2  # 共16个动作，编号 0~15
    save_path = os.path.join(save_dir, f'action_{action_idx}.npy')
    np.save(save_path, combined)

    print(f"已保存: action_{action_idx}.npy, 形状: {combined.shape}")
