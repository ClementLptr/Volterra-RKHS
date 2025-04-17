import torch
import torch.nn as nn
from config.logger import setup_logger
from network.base.base_vnn import BaseVNN

logger = setup_logger()

class VNN(BaseVNN):
    def __init__(self, num_classes, num_ch=3, pretrained=False):
        super(VNN, self).__init__()
        self.Q1 = 7
        self.nch_out1_5 = 8
        self.nch_out1_3 = 8
        self.nch_out1_1 = 8
        self.sum_chans = self.nch_out1_5 + self.nch_out1_3 + self.nch_out1_1

        self.Q2 = 7
        self.nch_out2 = 32

        self.Q3 = 7
        self.nch_out3 = 64

        self.Q4 = 7
        self.nch_out4 = 96

        self.Q5 = 7
        self.nch_out5 = 256

        self.conv11_5 = nn.Conv3d(num_ch, self.nch_out1_5, kernel_size=3, padding=1)
        self.conv11_3 = nn.Conv3d(num_ch, self.nch_out1_3, kernel_size=3, padding=1)
        self.conv11_1 = nn.Conv3d(num_ch, self.nch_out1_1, kernel_size=1)

        self.bn11 = nn.BatchNorm3d(self.sum_chans)

        self.conv21_5 = nn.Conv3d(num_ch, 2 * self.Q1 * self.nch_out1_5, kernel_size=3, padding=1)
        self.conv21_3 = nn.Conv3d(num_ch, 2 * self.Q1 * self.nch_out1_3, kernel_size=3, padding=1)
        self.conv21_1 = nn.Conv3d(num_ch, 2 * self.Q1 * self.nch_out1_1, kernel_size=1)

        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn21 = nn.BatchNorm3d(self.sum_chans)

        self.conv12 = nn.Conv3d(self.sum_chans, self.nch_out2, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm3d(self.nch_out2)
        self.conv22 = nn.Conv3d(self.sum_chans, 2 * self.Q2 * self.nch_out2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn22 = nn.BatchNorm3d(self.nch_out2)

        self.conv13 = nn.Conv3d(self.nch_out2, self.nch_out3, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm3d(self.nch_out3)
        self.conv23 = nn.Conv3d(self.nch_out2, 2 * self.Q3 * self.nch_out3, kernel_size=3, padding=1)
        self.bn23 = nn.BatchNorm3d(self.nch_out3)

        self.conv14 = nn.Conv3d(self.nch_out3, self.nch_out4, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm3d(self.nch_out4)
        self.conv24 = nn.Conv3d(self.nch_out3, 2 * self.Q4 * self.nch_out4, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn24 = nn.BatchNorm3d(self.nch_out4)

        self.conv15 = nn.Conv3d(self.nch_out4, self.nch_out5, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm3d(self.nch_out5)
        self.conv25 = nn.Conv3d(self.nch_out4, 2 * self.Q5 * self.nch_out5, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bn25 = nn.BatchNorm3d(self.nch_out5)

        self.fc1 = nn.Linear(12544, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.__init_weight()

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        return x

    def _forward_features(self, x):
        x11_5 = self.conv11_5(x)
        x11_3 = self.conv11_3(x)
        x11_1 = self.conv11_1(x)
        x11 = torch.cat((x11_5, x11_3, x11_1), 1)
        x11 = self.bn11(x11)

        x21_5 = self.conv21_5(x)
        x21_5add = self._volterra_kernel_approximation(x11_5, x21_5, self.Q1, self.nch_out1_5)

        x21_3 = self.conv21_3(x)
        x21_3add = self._volterra_kernel_approximation(x11_3, x21_3, self.Q1, self.nch_out1_3)

        x21_1 = self.conv21_1(x)
        x21_1add = self._volterra_kernel_approximation(x11_1, x21_1, self.Q1, self.nch_out1_1)

        x21_add = torch.cat((x21_5add, x21_3add, x21_1add), 1)
        x21_add = self.bn21(x21_add)

        x = self.pool1(x11 + x21_add)

        x12 = self.conv12(x)
        x12 = self.bn12(x12)
        x22 = self.conv22(x)
        x22_add = self._volterra_kernel_approximation(x12, x22, self.Q2, self.nch_out2)
        x22_add = self.bn22(x22_add)
        x = self.pool2(x12 + x22_add)

        x13 = self.conv13(x)
        x13 = self.bn13(x13)
        x23 = self.conv23(x)
        x23_add = self._volterra_kernel_approximation(x13, x23, self.Q3, self.nch_out3)
        x23_add = self.bn23(x23_add)
        x = x13 + x23_add

        x14 = self.conv14(x)
        x14 = self.bn14(x14)
        x24 = self.conv24(x)
        x24_add = self._volterra_kernel_approximation(x14, x24, self.Q4, self.nch_out4)
        x24_add = self.bn24(x24_add)
        x = self.pool4(x14 + x24_add)

        x15 = self.conv15(x)
        x15 = self.bn15(x15)
        x25 = self.conv25(x)
        x25_add = self._volterra_kernel_approximation(x15, x25, self.Q5, self.nch_out5)
        x25_add = self.bn25(x25_add)
        x = self.pool5(x15 + x25_add)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv11_5, model.conv11_3, model.conv11_1, model.bn11, model.conv21_5, model.conv21_3, model.conv21_1, model.bn21, model.conv12, model.bn12, model.conv22, model.bn22] #, model.conv13, model.bn13, model.conv23, model.bn23, model.conv14, model.bn14, model.conv24, model.bn24, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k  

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = VNN(num_classes=51, pretrained=False)
    outputs = net.forward(inputs)
    print(outputs.size())  # Should output torch.Size([1, 51])