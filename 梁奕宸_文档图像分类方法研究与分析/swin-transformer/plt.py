import matplotlib.pyplot as plt
import re

# 定义文件路径
file_path = '"tmp\pycharm_project_762\swin-transformer\run_results\train_log_results.txt"'

# 初始化列表存储数据
train_losses = []
test_accuracies = []
learning_rates = []

# 正则表达式匹配需要的数据
loss_regex = r"训练损失:(\d+\.\d+)"
accuracy_regex = r"测试准确率:(\d+\.\d+)"
lr_regex = r"lr=(\d+\.\d+)"

# 读取文件并提取数据
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        loss_match = re.search(loss_regex, line)
        if loss_match:
            train_losses.append(float(loss_match.group(1)))

        accuracy_match = re.search(accuracy_regex, line)
        if accuracy_match:
            test_accuracies.append(float(accuracy_match.group(1)))

        lr_match = re.search(lr_regex, line)
        if lr_match:
            learning_rates.append(float(lr_match.group(1)))

# 绘制图表
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss', color='red')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='green')
plt.title('Test Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(learning_rates, label='Learning Rate', color='blue')
plt.title('Learning Rate Curve')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.show()
