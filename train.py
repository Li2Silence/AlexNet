"""
以下代码为模型训练需要导入的包
"""
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet

"""定义主函数"""


def main():
    epochs = 20  # 训练轮次
    batch_size = 16  # 每一批次加载多少张图像
    save_path = 'models/AlexNet.pth'  # 模型保存
    best_acc = 0.0  # 初始化最好的准确率
    image_path = r'../data'  # 数据集存放位置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 定义用什么设备训练，有GPU就用GPU，没有就用CPU

    """
    下面代码是为了将图像转为Tensor格式，并对图像进行了增强
    """
    data_transform = {
        "train": transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    """
    定义训练数据集
    """
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    """设置线程数量"""
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    """定义训练集的dataloader"""
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)

    """定义测试集的dataloader"""
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = AlexNet(num_classes=5, init_weights=False)  # 对神经网络进行实例化
    net.to(device)  # 把神经网络传入要使用的设备上面

    loss_function = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  # 定义优化器

    train_steps = len(train_loader)
    for epoch in range(epochs):
        net.train()  # 定义模型训练，启用BatchNormalization 和 Dropout
        running_loss = 0.0  # 初始化loss值
        train_bar = tqdm(train_loader, file=sys.stdout)  # 这一个函数的作用是显示进度条
        for step, data in enumerate(train_bar):
            images, labels = data
            """
            以下5行代码不能改变位置，任何网络几乎都要这么写。
            """
            optimizer.zero_grad()  # 将梯度置为0，网络参量进行反馈时，梯度是被积累的而不是被替换掉，但在每一个batch中，不需要将两个batch的梯度混合起来								   累积，所以在这里将每一个batch的梯度置为0.
            outputs = net(images.to(device))  # 将图像输入神经网络
            loss = loss_function(outputs, labels.to(device))  # 计算真实图像与预测图像的loss
            loss.backward()  # 将loss返回给神经网络，并根据loss优化每一层的参数
            optimizer.step()  # 通过梯度下降执行一步参数更新

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        net.eval()  # 定义模型测试，不启用BatchNormalization 和 Dropout
        acc = 0.0
        with torch.no_grad():  # 测试的时候，不需要进行梯度更新，所以要使用这个函数。
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num  # 计算每一轮模型的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        """
        如果这一轮的准确率高于以前的，就保存该轮的模型权重。
        """
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
