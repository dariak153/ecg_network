import torch
import torch.nn as nn
import torch.nn.functional as F

class Block1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.2, downsample=None):
        super(Block1D, self).__init__()
    
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.dropout(out)
        return out

def make_layer(in_channels, out_channels, num_blocks, stride, dropout):
    downsample = None
    if stride != 1 or in_channels != out_channels * Block1D.expansion:
        downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * Block1D.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels * Block1D.expansion)
        )
    layers = [Block1D(in_channels, out_channels, kernel_size=7, stride=stride, dropout=dropout, downsample=downsample)]
    for _ in range(1, num_blocks):
        layers.append(Block1D(out_channels, out_channels, kernel_size=7, stride=1, dropout=dropout))
    return nn.Sequential(*layers)

class Model(nn.Module):
    def __init__(self, input_channels=1, dropout=0.2, lstm_hidden1=64, lstm_hidden2=32,
                 fc_hidden=32, num_classes=1, output_length=5000):
        super(Model, self).__init__()

        self.output_length = output_length
        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = make_layer(64, 64, num_blocks=2, stride=1, dropout=dropout)
        self.layer2 = make_layer(64, 128, num_blocks=2, stride=2, dropout=dropout)
        self.layer3 = make_layer(128, 128, num_blocks=2, stride=1, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=lstm_hidden1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=lstm_hidden1 * 2, hidden_size=lstm_hidden2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden2 * 2, fc_hidden)
        self.fc_out = nn.Linear(fc_hidden, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.initial(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)

        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=False)
        x = x.transpose(1, 2)
        return torch.sigmoid(x)
