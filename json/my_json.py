import os
import shutil
import json

def load_class_labels(class_file):
    with open(class_file, 'rb') as f:  # 以二进制模式打开文件
        class_data = json.load(f, encoding='utf-8')  # 使用utf-8编码解析JSON数据
    return class_data

def organize_images(image_folder, class_labels):
    # 遍历文件夹中的文件
    for filename in os.listdir(image_folder):
        if filename.endswith('.json'):
            # 读取JSON文件中的标签信息
            with open(os.path.join(image_folder, filename), 'r') as f:
                json_data = json.load(f)
            label = class_labels.get(filename[:-5])  # 移除'.json'后缀
            if label:
                # 新建类别文件夹
                label_folder = os.path.join(image_folder, label)
                os.makedirs(label_folder, exist_ok=True)
                # 将图片移动到对应类别文件夹
                shutil.move(os.path.join(image_folder, filename[:-5] + '.png'), label_folder)

if __name__ == "__main__":
    # 文件夹路径和文件名
    image_folder = 'D:/Desktop/division/NKU页面类型/total'
    class_file = 'D:/Desktop/division/NKU页面类型/total/class.json'

    # 加载类别标签
    class_labels = load_class_labels(class_file)

    # 对图片进行分类
    organize_images(image_folder, class_labels)
