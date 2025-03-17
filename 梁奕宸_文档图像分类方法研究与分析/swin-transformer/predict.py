import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torchvision import transforms
import numpy as np
from model import swin_base_patch4_window12_384_22k as svit
# from model import swin_base_patch4_window7_224_22k as svit
import cv2
import json
from PIL import Image

def main(num_classes):
    # 预处理
    data_transform = transforms.Compose([transforms.Resize(448),transforms.CenterCrop(384),transforms.ToTensor(),
                                         transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

    # get devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model
    model = svit(num_classes=num_classes)
    model.load_state_dict(torch.load("./run_results/best_model.pth")) # 载入best_model进行预测，上传至device
    model.to(device)

    # json 文件
    try:
        json_file = open('class_indices.json', 'r') # 从json文件中读取类别标签
        labels = json.load(json_file)
    except Exception as e:
        print(e)

    # load image
    model.eval()  # 进入验证模式
    test_path = './inference'
    test_imgs = [os.path.join(test_path, i) for i in os.listdir(test_path) if not i.startswith('.')]
    with torch.no_grad():
        for test_img in test_imgs:
            original_img = Image.open(test_img).convert('RGB')
            src = cv2.cvtColor(np.asarray(original_img), cv2.COLOR_BGR2RGB)
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)
            output = model(img.to(device))
            output = torch.softmax(output,dim=1)
            p,index = torch.topk(output,k=2)            # 保留前三个概率
            p = p.to("cpu").numpy()[0]
            index = index.to("cpu").numpy()[0]

            # 打印前三个类别
            for i,(x,y) in enumerate(zip(p,index)):
                y = labels[str(y)]
                text = '%s :%.4f' % (y,x)
                cv2.putText(src,text,(10,10+20*(i+1)), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)    # 输入文本
                a = test_img.split('.')[-2]
                save_path ='.'+a + '_result.' + test_img.split('.')[-1]
                cv2.imwrite(save_path,src)              # 保存结果


if __name__ == '__main__':
    # 根据分类任务更改
    num_classes = 9
    main(num_classes=num_classes)
