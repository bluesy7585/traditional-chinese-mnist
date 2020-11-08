import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):

    def __init__(self, nc, nclass):
        super(ConvNet, self).__init__()
        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        #nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [64, 128, 256, 256, 256, 128, 64]
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))

            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0) # 1x48x16 > 64x48x16
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        # 64x48x16 > 64x24x8
        convRelu(1) # 64x24x8 > 128x24x8
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        # 128x12x4
        convRelu(2, True) # 256x12x4
        convRelu(3) # 256x12x4
        # Cx12x4

        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 2)))

        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 2)))

        self.cnn = cnn
        self.linear = nn.Linear(4608, 512)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(512, nclass)

    def num_flat_features(self, x):
        #print(x.size()) # batch size x 50 x 4 x 4
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        x = conv.view(-1, self.num_flat_features(conv))
        #print(x.size())
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.linear2(x)
        return output

    def name(self):
        return "ConvNet"
