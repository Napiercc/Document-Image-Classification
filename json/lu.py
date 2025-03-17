import os
import json
import shutil

# 假设所有文件都在当前目录下
current_directory = 'D:\\Desktop\\division\\300_book\\total\\total'
class_file_path = os.path.join(current_directory, 'result.json')

# 指定分类文件夹的位置
category_folder_base = 'C:\\temp'

# 读取class.json文件
with open(class_file_path, 'r') as file:
    class_info = json.load(file)

# 遍历class_info中的每一项，按类别组织文件
for json_file, category in class_info.items():
    # 如果文件不是json文件，跳过
    if not json_file.endswith('.json'):
        continue

    # 创建对应类别的文件夹（如果不存在）
    category_folder_path = os.path.join(category_folder_base, category)
    if not os.path.exists(category_folder_path):
        os.makedirs(category_folder_path)

    # 对应的图片文件名（假设图片文件名与json文件名相同，只是后缀不同）
    image_file = os.path.splitext(json_file)[0] + '.jpg'
    image_file_path = os.path.join(current_directory, image_file)

    # 如果图片文件存在，则复制到对应的类别文件夹
    if os.path.exists(image_file_path):
        shutil.copy(image_file_path, category_folder_path)
    else:
        print(f"文件 {image_file} 不存在。")

    # 输出每个类别的名称
    print(f"类别: {category}")

print("分类完成。")
