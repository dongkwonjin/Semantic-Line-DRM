import torch
import torch.nn as nn
import torch.nn.functional as F


class Fully_connected_layer(nn.Module):
    def __init__(self):
        super(Fully_connected_layer, self).__init__()

        self.linear_1 = nn.Linear(2048, 1024)
        self.linear_2 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        fc1 = self.linear_1(x)
        fc1 = F.relu(fc1)
        fc2 = self.linear_2(fc1)
        fc2 = F.relu(fc2)

        return fc2


class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()

        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()

        self.fully_connected = Fully_connected_layer()
        self.classification = Classification()

    def forward(self, feat1, feat2, is_training=True):

        f_concat = torch.cat((feat1['fc1'], feat2['fc1']), dim=1)
        fc_out = self.fully_connected(f_concat)
        cls = self.classification(fc_out)

        if is_training is True:
            return {'cls': cls}
        else:
            return {'cls': cls}

class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()

        self.fully_connected = Fully_connected_layer()
        self.classification = Classification()

    def forward(self, feat1, feat2, is_training=True):

        f_concat = torch.cat((feat1['fc1'], feat2['fc1']), dim=1)
        fc_out = self.fully_connected(f_concat)
        cls = self.classification(fc_out)

        if is_training is True:
            return {'cls': cls}
        else:
            return {'cls': cls}