"""
作业2 - 任务2-1：音素分类 —— 参考代码 (reference.py)
来源：Colab HW02-1.ipynb 示例代码
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import csv

myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

DATA_DIR = "../data/timit_11/timit_11"

print("Loading data ...")
train = np.load(f"{DATA_DIR}/train_11.npy")
train_label = np.load(f"{DATA_DIR}/train_label_11.npy").astype(np.int64)
test = np.load(f"{DATA_DIR}/test_11.npy")
print(f"Size of training data: {train.shape}")
print(f"Size of testing data: {test.shape}")


class TIMITDataset(Dataset):
    """TIMIT 音素分类数据集"""

    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        return self.data[idx]

    def __len__(self):
        return len(self.data)


VAL_RATIO = 0.2
percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y = train[:percent], train_label[:percent]
val_x, val_y = train[percent:], train_label[percent:]
print(f"Size of training set: {train_x.shape}")
print(f"Size of validation set: {val_x.shape}")

BATCH_SIZE = 64
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)


class Classifier(nn.Module):
    """参考模型：4层 MLP，每层含 Linear + BatchNorm + ReLU + Dropout"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.layer0 = nn.Linear(429, 2048)
        self.layer1 = nn.Linear(2048, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39)

        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        self.batchnorm0 = nn.BatchNorm1d(2048)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.batchnorm3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.layer0(x)
        x = self.batchnorm0(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.batchnorm1(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.batchnorm2(x)
        x = self.act_fn(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.batchnorm3(x)
        x = self.act_fn(x)
        x = self.out(x)
        return x


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
print(f"DEVICE: {device}")

num_epoch = 40
learning_rate = 0.0001
model_path = "./model.ckpt"

model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.NADM(model.parameters(), lr=learning_rate)

best_acc = 0.0

print("\n开始训练...")
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1)

        batch_loss.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    if len(val_set) > 0:
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()

        print(
            f"[{epoch + 1:03d}/{num_epoch:03d}] "
            f"Train Acc: {train_acc / len(train_set):.6f} "
            f"Loss: {train_loss / len(train_loader):.6f} | "
            f"Val Acc: {val_acc / len(val_set):.6f} "
            f"Loss: {val_loss / len(val_loader):.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  saving model with acc {best_acc / len(val_set):.3f}")
    else:
        print(
            f"[{epoch + 1:03d}/{num_epoch:03d}] "
            f"Train Acc: {train_acc / len(train_set):.6f} "
            f"Loss: {train_loss / len(train_loader):.6f}"
        )
        torch.save(model.state_dict(), model_path)

print(f"\n最佳验证准确率: {best_acc / len(val_set):.4f}")

# ==================== 预测 ====================
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

predict = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1)
        predict.extend(test_pred.cpu().numpy().tolist())

with open("prediction.csv", "w") as f:
    f.write("Id,Class\n")
    for i, y in enumerate(predict):
        f.write(f"{i},{y}\n")

print("预测完成，结果已保存至 prediction.csv")
