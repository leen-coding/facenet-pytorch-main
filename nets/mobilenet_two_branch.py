import torch
import torch.nn as nn


# updates
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()
        # nn.PReLU()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam(nn.Module):
    def __init__(self, planes, ratio):
        super(cbam, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size=5)

    def forward(self, x):
        # ca_map = self.ca(x)
        x = self.ca(x) * x  # 广播机制
        # sa_map = self.sa(x)
        x = self.sa(x) * x  # 广播机制
        return x


def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )

def conv_dw_branch(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )

class testmodel(nn.Module):
    def __init__(self):
        super(testmodel, self).__init__()
        # 160,160,3 -> 80,80,32
        self.c1 = conv_bn(3, 32, 2)
        # 80,80,32 -> 80,80,64
        self.c2 = conv_dw(32, 64, 1)

        # 80,80,64 -> 40,40,128
        self.c3 = conv_dw(64, 128, 2)
        self.c4 = conv_dw(128, 128, 1)

        # 40,40,128 -> 20,20,256
        self.c5 = conv_dw(128, 256, 2)
        self.c6 = conv_dw(256, 256, 1)

        self.c7 =conv_dw(256, 512, 2)
        self.c8 =conv_dw(512, 512, 1)
        self.c9 =conv_dw(512, 512, 1)
        self.c10 =conv_dw(512, 512, 1)
        self.c11 =conv_dw(512, 512, 1)
        self.c12 =conv_dw(512, 512, 1)

        self.c13 =conv_dw(512, 1024, 2)
        self.c14 =conv_dw(1024, 1024, 1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)


        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.c11(x)
        x = self.c12(x)


        x = self.c13(x)
        x = self.c14(x)

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(3, 32, 2),
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1),

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.cbam_fun = cbam(1024, 16)

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.cbam_fun(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = testmodel()

    x = torch.zeros([2,3,160,160])
    x = model(x)
    print("test")
