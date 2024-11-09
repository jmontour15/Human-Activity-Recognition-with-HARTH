from torch import nn

class HAR_Hybrid(nn.Module):
    def __init__(self, num_classes=6, input_channels=6, sequence_length=50, dropout_prob=0.3):
        super(HAR_Hybrid, self).__init__()

        # Conv layer 1
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Conv layer 2
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        # Conv Layer 3
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        # Maxpool layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def forward(self, x):
        # Apply conv -> batch norm -> ReLU -> max pooling
        x = self.pool(nn.ReLU()(self.bn1(self.conv1(x))))
        x = self.pool(nn.ReLU()(self.bn2(self.conv2(x))))
        
        # Transpose for LSTM input (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

        # LSTM processing
        _, (x, _) = self.lstm(x)
        
        # Squeeze out the sequence dimension after LSTM
        x = x.squeeze(0)
        
        # Apply fully connected layers with dropout
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)  # Output layer
        
        return x
