"""
以下代码是模型预测需要引入的包
"""
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_path = r"F:\Zhongxing\经典论文\论文复现\code\pytorch_classification\data\val\roses\909277823_e6fb8cb5c8_n.jpg" # 要测试的图像
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path) # 读取图像

    plt.imshow(img)
    img = data_transform(img) # 把图像转化为pytorch能处理的格式，此时图像为[C,H,W]
    img = torch.unsqueeze(img, dim=0) # 在训练时，图像的格式为[B,C,H,W]，而现在图像的格式为[C,H,W]，所以需要对图像添加一维。

    json_path = './class_indices.json' # 训练时，保存的类别文本文档
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = AlexNet(num_classes=5).to(device)

    weights_path = "models/AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path)) # 加载训练模型，将模型权重放入神经网络。

    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
