from torch import nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


# 基于res net 的模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = conv_block(3, 32)  # 3,32,32
        self.conv2 = conv_block(32, 64, pool=True)  # 64,16,16
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64))  # 64, 16, 16
        self.conv3 = conv_block(64, 128)  # 128, 16, 16
        self.conv4 = conv_block(128, 256, pool=True)  # 256, 8, 8
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))  # 256,8,8
        self.conv5 = conv_block(256, 512)  # 512, 8, 8
        self.conv6 = conv_block(512, 1024, pool=True)  # 1024, 4, 4
        self.res3 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024))  # 1024, 4, 4
        self.linear1 = nn.Sequential(nn.MaxPool2d(4),  # 1024,1,1
                                     nn.Flatten(),
                                     nn.Dropout(0.2),
                                     nn.Linear(1024, 10))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.res3(out) + out
        out = self.linear1(out)

        return out
