import os
import math
import argparse
import shutil
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, datasets
# from model import swin_base_patch4_window12_384_22k as svit
from model import swin_base_patch4_window7_224_22k as svit
# from model import swin_tiny_patch4_window7_224_22k as svit
from utils import (train_one_epoch, evaluate,plt_loss_acc)


def main(args):
    # 网络的训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device))

    with open('./run_results/train_log_results.txt', "a") as f:  # 保存训练信息, a --> 在文件中追加信息
        info = f"[train hyper-parameters: {args}]\n\n"
        f.write(info)

    # 预处理
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),       # 水平翻转
                                          transforms.ToTensor(),
                                           transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    test_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                         transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    # 计算加载的线程数
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % num_workers)

    # 加载训练集
    train_dataset = datasets.ImageFolder(root='/tmp/pycharm_project_762/swin-transformer/data/train',
                                         transform=train_transform)
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
    weights_dict = torch.load('swin_base_patch4_window7_224_22k.pth', map_location=device)['model']

    # 删除不需要的权重
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]
    net.load_state_dict(weights_dict, strict=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-8)
    net.to(device)

    # 自适应学习率衰减
    lf = lambda x: ((1 + math.cos(x * math.pi / 100)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0
    train_loss_list = []  # 训练集的损失
    test_acc_list = []  # 测试集的精度
    lr_list = []
    for epoch in range(100):
        train_loss, lr = train_one_epoch(model=net, optim=optimizer, train_loader=train_loader,
                                         device=device, num_train=num_trainset)
        scheduler.step()

        test_acc = evaluate(model=net, test_loader=test_loader, device=device, num_test=num_testset)

        # 记录训练集和测试集的信息
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        lr_list.append(lr)

        # 保存训练信息, a --> 在文件中追加信息
        with open('./run_results/train_log_results.txt', "a") as f:
            info = f"[epoch: {epoch + 1}]\n" + f"train loss:{train_loss:.4f}\t" + f"test accuracy:{test_acc:.4f}\n\n"
            f.write(info)

        if test_acc > best_acc:  # 保留最好的权重
            best_acc = test_acc
            torch.save(net.state_dict(), './run_results/best_model.pth')

        # 控制台的打印信息
        print("[epoch:%d]" % (epoch + 1))
        print("learning rate:%.8f" % lr)
        print("train loss:%.4f" % train_loss)
        print("test accuracy:%.4f" % test_acc, end='\n')

    print('Training over !!!')

    # 绘制loss和accuracy曲线
    plt_loss_acc(train_loss_list, test_acc_list, lr_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="image classification")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lrf', default=0.01, type=float)
    # 最终学习率 = lr * lrf

    args = parser.parse_args()
    print(args)

    # 删除上次保留权重和训练日志，重新创建
    if os.path.exists("./run_results"):
        shutil.rmtree('./run_results')
    os.mkdir("./run_results")

    main(args)
