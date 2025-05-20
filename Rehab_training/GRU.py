from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 修改为你的npy路径 ====
X = np.load(r'D:\pycharm project\Rehab-120\Rehab_training\data\X.npy').astype(np.float32)  # shape: (83088, 200, 12)
y = np.load(r'D:\pycharm project\Rehab-120\Rehab_training\data\y.npy').astype(np.int64)    # shape: (83088,)

# ==== 数据集准备 ====
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==== 模型定义 ====
class BiGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)  # out: (batch, seq_len, 2*hidden_size)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out

# ==== 参数配置 ====
input_size = 12
hidden_size = 128
num_layers = 2
num_classes = 16
num_epochs = 20

model = BiGRUClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==== 训练 ====
best_val_acc = 0.0
best_model_path = r'D:\pycharm project\Rehab-120\Rehab_training\models\best_GRU_model.pth'

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = batch_y.cpu().numpy()

        train_preds.extend(preds)
        train_labels.extend(labels)

        # 每个 batch 打印
        batch_acc = accuracy_score(labels, preds)
        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Batch {batch_idx + 1}/{len(train_loader)} | "
              f"Loss: {loss.item():.4f} | "
              f"Batch Acc: {batch_acc:.4f}")

    train_acc = accuracy_score(train_labels, train_preds)

    # ==== 验证 ====
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            val_labels.extend(batch_y.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"===> Epoch {epoch + 1} Summary | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"--> [Epoch {epoch + 1}] New best model saved with Val Acc: {val_acc:.4f}")

# ==== 加载最优模型 & 评估 ====
print("\n=== Evaluating Best Model ===")
model.load_state_dict(torch.load(best_model_path))
model.eval()

val_preds, val_labels = [], []
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        val_labels.extend(batch_y.cpu().numpy())

acc = accuracy_score(val_labels, val_preds)
precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
cm = confusion_matrix(val_labels, val_preds)

print(f"\nFinal Evaluation:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nConfusion Matrix:\n", cm)

# ==== 绘制混淆矩阵 ====
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
