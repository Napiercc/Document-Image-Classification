import os
import random
import shutil

# 指定数据集所在的文件夹路径
data_folder = 'C:\\temp'

# 指定训练集、验证集和测试集的比例
train_ratio = 0.8  # 训练集占比60%
# val_ratio = 0.2    # 验证集占比20%
test_ratio = 0.2   # 测试集占比20%

# 确定训练集、验证集和测试集的文件夹路径
train_folder = 'C:\\temp\\300_train'
# val_folder = 'D:\\Desktop\\division\\swin-transformer\\swin-transformer\\data\\val1'
test_folder = 'C:\\temp\\300_test'

# 如果训练集、验证集和测试集文件夹不存在，则创建它们
for folder in [train_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 遍历每个类别文件夹
for category_folder in os.listdir(data_folder):
    category_folder_path = os.path.join(data_folder, category_folder)
    if os.path.isdir(category_folder_path):
        # 确定每个类别的训练集、验证集和测试集文件夹路径
        train_category_folder = os.path.join(train_folder, category_folder)
        # val_category_folder = os.path.join(val_folder, category_folder)
        test_category_folder = os.path.join(test_folder, category_folder)
        if not os.path.exists(train_category_folder):
            os.makedirs(train_category_folder)
        # if not os.path.exists(val_category_folder):
        #     os.makedirs(val_category_folder)
        if not os.path.exists(test_category_folder):
            os.makedirs(test_category_folder)

        # 遍历类别文件夹中的文件，并随机划分到训练集、验证集和测试集
        for filename in os.listdir(category_folder_path):
            file_path = os.path.join(category_folder_path, filename)
            if os.path.isfile(file_path):
                # 随机确定该文件划分到训练集、验证集还是测试集
                rand = random.random()
                if rand < train_ratio:
                    # 将文件移动到训练集文件夹
                    shutil.copy(file_path, os.path.join(train_category_folder, filename))
                # elif rand < train_ratio + val_ratio:
                #     # 将文件移动到验证集文件夹
                #     shutil.copy(file_path, os.path.join(val_category_folder, filename))
                else:
                    # 将文件移动到测试集文件夹
                    shutil.copy(file_path, os.path.join(test_category_folder, filename))

print("数据集划分完成。")
