import os
import math
import argparse
import shutil
import time
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
from model import swin_base_patch4_window12_384_22k as svit
from utils import (train_one_epoch, evaluate, plt_loss_acc)


def main(args):
    start_time = time.time()  # 记录开始时间

    # 网络的训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用 {} 进行训练.".format(device))

    # 保存训练信息
    with open('./run_results/train_log_results.txt', "a") as f:
        info = f"[训练超参数: {args}]\n\n"
        f.write(info)

    # 预处理
    train_transform = transforms.Compose([transforms.Resize(512),
                                          transforms.CenterCrop(384),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    test_transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(384),
                                         transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 计算加载的线程数
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])
    print('使用 %g 个数据加载器线程' % num_workers)

    # 加载训练集
    train_dataset = datasets.ImageFolder(root='/tmp/pycharm_project_762/swin-transformer/data/train',
                                         transform=train_transform)

    # 仅提取目标类别的样本
    target_class_name = 'BodyPage_TableTitle'
    target_class_idx = train_dataset.class_to_idx[target_class_name]
    target_class_data = [sample for sample in train_dataset.samples if sample[1] == target_class_idx]
    train_dataset.samples = target_class_data

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers,
                                               shuffle=True)

    # 加载测试集
    test_dataset = datasets.ImageFolder(root='/tmp/pycharm_project_762/swin-transformer/data/test',
                                        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    # 数据集个数
    num_trainset = len(train_dataset)
    num_testset = len(test_dataset)

    # 加载模型，自动导入预训练权重
    net = svit(num_classes=args.num_classes).to(device)
    weights_dict = torch.load('swin_base_patch4_window12_384_22k.pth', map_location=device)['model']

    # 删除不需要的权重
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]
    net.load_state_dict(weights_dict, strict=False)

    # 加载之前的最佳模型权重
    best_model_path = './run_results/best_model.pth'
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path, map_location=device))
        print("加载最佳模型权重: {}".format(best_model_path))
    else:
        print("未找到最佳模型权重，使用预训练权重进行训练。")

    # 冻结底层网络参数
    for param in net.parameters():
        param.requires_grad = False
    # 解冻最后一层分类层的参数
    for param in net.head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-8)
    net.to(device)

    # 自适应学习率衰减
    lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    best_epoch = 0  # 记录最好的权重对应的轮次
    train_loss_list = []  # 训练集的损失
    test_acc_list = []  # 测试集的精度
    lr_list = []

    # 初始化一个字典来存储每个类别正确预测的数量
    class_correct = {classname: 0 for classname in train_dataset.classes}
    # 初始化一个字典来存储每个类别的样本总数
    class_total = {classname: 0 for classname in train_dataset.classes}

    # 记录显存使用情况
    start_memory = torch.cuda.memory_allocated()
    max_memory = torch.cuda.max_memory_allocated()

    for epoch in range(100):
        epoch_start_time = time.time()  # 记录每个epoch的开始时间

        train_loss, lr = train_one_epoch(model=net, optim=optimizer, train_loader=train_loader,
                                         device=device, num_train=num_trainset)
        scheduler.step()

        test_acc, class_correct, class_total = evaluate(model=net, test_loader=test_loader, device=device,
                                                        num_test=num_testset,
                                                        class_correct=class_correct, class_total=class_total)

        epoch_end_time = time.time()  # 记录每个epoch的结束时间

        # 记录训练集和测试集的信息
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        lr_list.append(lr)

        # 保存训练信息, a --> 在文件中追加信息
        with open('./run_results/train_log_results.txt', "a") as f:
            info = f"[第 {epoch + 1} 轮]\n" + f"训练损失:{train_loss:.4f}\t" + f"测试准确率:{test_acc:.4f}\n\n"
            f.write(info)

        if test_acc > best_acc:  # 保留最好的权重
            best_acc = test_acc
            best_epoch = epoch + 1  # 更新最好的权重对应的轮次
            torch.save(net.state_dict(), './run_results/best_model_2.pth')

        # 控制台的打印信息
        print("[第%d轮]" % (epoch + 1))
        print("学习率:%.8f" % lr)
        print("训练损失:%.4f" % train_loss)
        print("测试准确率:%.4f" % test_acc)
        print("本轮次训练时间: {:.2f} 秒".format(epoch_end_time - epoch_start_time))  # 打印每个epoch的训练时间

        # 更新显存使用情况
        start_memory = max(start_memory, torch.cuda.memory_allocated())
        max_memory = max(max_memory, torch.cuda.max_memory_allocated())

    print('训练结束 !!!')
    print('最好的权重对应的轮次为:', best_epoch)

    # 计算并打印每个类别的测试准确率
    for classname, correct_count in class_correct.items():
        accuracy = 100 * correct_count / class_total[classname]
        print("类别 {} 的测试准确率: {:.2f}%".format(classname, accuracy))

    # 绘制loss和accuracy曲线
    plt_loss_acc(train_loss_list, test_acc_list, lr_list)

    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算代码运行时间
    print("代码运行时间: {:.2f} 秒".format(execution_time))

    # 打印显存使用情况
    print("最大显存使用量: {:.2f} MB".format(max_memory / (1024 ** 2)))
    print("训练期间显存使用量: {:.2f} MB".format(start_memory / (1024 ** 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="图像分类")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lrf', default=0.01, type=float)
    # 最终学习率 = lr * lrf

    args = parser.parse_args()
    print(args)

    # # 删除上次保留权重和训练日志，重新创建
    # if os.path.exists("./run_results"):
    #     shutil.rmtree('./run_results')
    # os.mkdir("./run_results")

    # 检查并创建run_results文件夹，如果不存在则创建
    if not os.path.exists("./run_results"):
        os.mkdir("./run_results")
    main(args)
