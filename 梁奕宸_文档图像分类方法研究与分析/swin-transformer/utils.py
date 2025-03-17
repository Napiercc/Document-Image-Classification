import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


# 训练一个 epoch
def train_one_epoch(model, optim,train_loader, device,num_train):
    loss_function = torch.nn.CrossEntropyLoss()         # 损失函数
    model.train()
    train_running_loss = 0.0        # 训练集的总损失
    for train_image, train_target in tqdm(train_loader):
        train_image, train_target = train_image.to(device), train_target.to(device)
        train_output = model(train_image)                   # 前向传播
        loss = loss_function(train_output, train_target)    # 计算损失
        optim.zero_grad()                   # 梯度清零
        loss.backward()                     # 反向传播
        optim.step()                        # 梯度更新
        train_running_loss += loss.item()

    lr = optim.param_groups[0]["lr"]
    return train_running_loss/num_train,lr


# 计算性能指标
def evaluate(model, test_loader, device, num_test, class_correct=None, class_total=None):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    if class_correct is None:
        class_correct = {classname: 0 for classname in test_loader.dataset.classes}
    if class_total is None:
        class_total = {classname: 0 for classname in test_loader.dataset.classes}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算每个类别的正确预测数量和样本总数
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[test_loader.dataset.classes[label]] += (predicted[i] == labels[i]).item()
                class_total[test_loader.dataset.classes[label]] += 1

    test_accuracy = correct / total
    return test_accuracy, class_correct, class_total



# 绘制loss 和 acc曲线
def plt_loss_acc(train_loss,test_acc,lr):
    plt.figure(figsize=(12,8))
    plt.subplot(1,3,1)
    plt.plot(train_loss,label='train loss',linestyle='-',color='r')
    plt.title('loss curve')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(test_acc,label='test accuracy',linestyle='-',color='g')
    plt.title('accuracy curve')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(lr,label='lr',linestyle='-',color='b')
    plt.title('learning rate curve')
    plt.legend()

    plt.savefig('./run_results/loss_accuracy_lr.png',dpi=300)
    plt.clf()
