from torch import nn

class HAR_CNN_1D(nn.Module):
    def __init__(self, num_classes=6, input_channels=6, sequence_length=50, dropout_prob=0.2):
        super(HAR_CNN_1D, self).__init__()

        # Conv layer 1
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)

        # Conv layer 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Conv layer 3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        # Maxpool layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Compute the size of the output after convolutional layers and pooling
        conv_output_size = sequence_length // 8  # After 3 pooling layers with stride 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Apply first conv -> batch norm -> ReLU -> max pooling
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        
        # Apply second conv -> batch norm -> ReLU -> max pooling
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        
        # Apply third conv -> batch norm -> ReLU -> max pooling
        x = self.pool(nn.ReLU()(self.bn3(self.conv3(x))))
        
        # Flatten the tensor before passing into fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with dropout
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)  # Dropout after the first fully connected layer
        
        x = self.fc2(x)  # No activation on the final output (will be handled by the loss function)
        
        return x
    