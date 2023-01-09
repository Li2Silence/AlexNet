"""以下两行代码定义要引入的包"""
import torch.nn as nn
import torch

"""
下面为定义网络结构
nn.Conv2d(3, 48, kernel_size=(11,11), stride=(4,4), padding=2) 这是定义卷积层，参数分别为输入通道大小，输出通道大小，卷积核尺寸，步长大小，填充大小
nn.ReLU(inplace=True) 定义Relu激活函数
nn.MaxPool2d(kernel_size=3, stride=2) 定义最大池化层，kernel_size > stride为重叠池化

x = torch.flatten(x, start_dim=1) 表示将特征图拉伸成一维的长向量，然后输入全连接层

nn.Linear(2048, 2048)  这是全连接层的定义方法，参数分别为输入的向量维度，输出的向量维度
nn.Dropout(p=0.5)  dropout层，因为全连接层有很多的冗余，防止模型过拟合
"""
class AlexNet(nn.Module):
    def __init__(self, num_classes=5, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(11,11), stride=(4,4), padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=(5,5), padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=(3,3), padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=(3,3), padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=(3,3), padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    """
    以下代码为模型初始化，可以不用。
    """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
