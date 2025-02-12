import torch as th
import torch.nn as nn

#############################
#############################
#############################
# FNN Model


class NIDSFNN(nn.Module):
    def __init__(self, hidden_units, num_features, num_classes, use_bn=True, dropout=0.5,
                 model_name="fnn"):
        """
        Args:
            hidden_units (list): A list of integers specifying the number of neurons in each hidden layer.
                                 e.g., [128, 64] -> two layers with 128 and 64 neurons, respectively.
            num_features  (int): Number of input features per sample.
            num_classes   (int): Number of classes (for classification).
            use_bn       (bool): Whether to use Batch Normalization after each hidden layer.
            dropout     (float): Dropout probability in the classifier.
        """
        super().__init__()

        self.model_name = model_name

        layers = []
        input_dim = num_features

        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Final layer for classification
        layers.append(nn.Linear(input_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


#############################
#############################
#############################
# CNN Model


class NIDSCNN(nn.Module):
    def __init__(self, out_channels, num_features, num_classes, use_bn=True, dropout=0.5,
                 model_name="cnn"):
        """
        Args:
            out_channels (list): A list of integers specifying the out_channels for each Conv1D layer.
                                 e.g., [16, 32] -> two layers with 16 and 32 channels, respectively.
            num_features  (int): Number of input features per sample (length of the 1D input).
            num_classes   (int): Number of classes (for classification).
            use_bn       (bool): Whether to use Batch Normalization after each Conv1D.
            dropout     (float): Dropout probability in the classifier.
        """
        super().__init__()

        # super(NIDSCNN, self).__init__()
        self.model_name = model_name

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(64)

        in_features = 64 * ((num_features - 2) // 2)
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 1, input_dim)
        x = self.conv(x)
        x = th.relu(x)
        x = self.pool(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = th.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#############################
#############################
#############################
# GRU Model

class NIDSGRU(nn.Module):
    def __init__(
        self,
        num_features,   # Input size (number of features in each sample)
        hidden_size,    # Hidden size of the GRU
        num_layers,     # Number of GRU layers
        num_classes,    # Output classes for classification
        use_bn=False,   # Whether to use Batch Normalization
        # Dropout probability (applied in GRU if num_layers > 1)
        dropout=0.5,
        model_name="gru"
    ):
        # super(NIDSGRU, self).__init__()
        super().__init__()

        self.model_name = model_name
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bn = use_bn

        # GRU
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        if self.use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len=1, num_features)
        Output shape: (batch_size, num_classes)
        """
        # GRU forward
        # out: (batch_size, seq_len, hidden_size)
        # hn:  (num_layers, batch_size, hidden_size)
        out, hn = self.gru(x)

        # Take the last time step's output
        # If seq_len=1, out[:, -1, :] => out[:, 0, :]
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Apply BN if enabled
        if self.use_bn:
            out = self.bn(out)

        out = self.fc(out)
        out = self.fc2(out)
        return out


#############################
#############################
#############################
# LSTM Model

class NIDSLSTM(nn.Module):
    def __init__(
        self,
        num_features,   # Input size (number of features in each sample)
        hidden_size,    # Hidden size of the LSTM
        num_layers,     # Number of LSTM layers
        num_classes,    # Output classes for classification
        use_bn=False,   # Whether to use Batch Normalization
        # Dropout probability (applied in LSTM if num_layers > 1)
        dropout=0.5,
        model_name="lstm"
    ):
        # super(NIDSLSTM, self).__init__()
        super().__init__()

        self.model_name = model_name
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bn = use_bn

        # LSTM: We treat each sample as a sequence of length=1, with 'num_features' as input_size
        # batch_first=True => input shape: (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Optional BatchNorm on the final hidden state
        if self.use_bn:
            self.bn = nn.BatchNorm1d(hidden_size)

        # Final fully-connected layer
        self.fc = nn.Linear(hidden_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        """
        Input shape: (batch_size, seq_len=1, num_features)
        Output shape: (batch_size, num_classes)
        """
        # LSTM forward
        # out: (batch_size, seq_len, hidden_size)
        # (hn, cn): hidden and cell states with shape (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x)

        # We can take the output at the final timestep in the sequence.
        # If seq_len=1, out[:, -1, :] is simply out[:, 0, :] which is shape (batch_size, hidden_size)
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # If BN is used, apply on the final hidden vector
        if self.use_bn:
            out = self.bn(out)  # BN expects (batch_size, hidden_size)

        # Pass through fully-connected layer
        out = self.fc(out)
        out = self.fc2(out)
        return out


#############################
#############################
#############################
# CNN_LSTM Model

class CNN_LSTM(nn.Module):
    def __init__(
        self,
        # list[int]: channel sizes for each Conv1D layer, e.g. [16, 32]
        out_channels,
        seq_length,          # int: length of the input sequence
        num_classes,         # int: number of output classes
        cnn_use_bn=True,     # bool: whether to use BatchNorm in CNN
        lstm_hidden_size=64,  # int: hidden size of the LSTM
        lstm_num_layers=1,   # int: number of LSTM layers
        lstm_dropout=0.5,    # float: dropout in LSTM if lstm_num_layers > 1
        final_dropout=0.5,    # float: dropout before final classification layer
        model_name="cnn_lstm"
    ):
        # super(CNN_LSTM, self).__init__()
        super().__init__()
        self.model_name = model_name
        # ----------------------
        # 1) CNN feature extractor
        # ----------------------
        cnn_layers = []
        in_channels = 1  # input is (batch, 1, seq_length)

        for oc in out_channels:
            cnn_layers.append(
                nn.Conv1d(in_channels, oc, kernel_size=3, stride=1, padding=1))

            if cnn_use_bn:
                cnn_layers.append(nn.BatchNorm1d(oc))

            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(kernel_size=2))

            in_channels = oc

        self.cnn = nn.Sequential(*cnn_layers)

        # After each MaxPool1d(kernel_size=2), the sequence length is halved.
        # final_seq_len = seq_length / 2^(number_of_pools)
        self.final_seq_len = seq_length // (2 ** len(out_channels))

        # ----------------------
        # 2) LSTM
        # ----------------------
        # The LSTM input size = out_channels[-1]
        # We'll feed the CNN output (batch, out_channels[-1], final_seq_len)
        # into an LSTM whose input is (batch, final_seq_len, out_channels[-1])
        # => we will transpose before feeding into LSTM.
        self.lstm = nn.LSTM(
            input_size=out_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            # input/output shape: (batch, seq_len, input_size)
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0
        )

        # ----------------------
        # 3) Classifier
        # ----------------------
        self.dropout = nn.Dropout(final_dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, 1, seq_length)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # --- CNN ---
        # shape after CNN: (batch_size, out_channels[-1], final_seq_len)
        x = self.cnn(x)

        # Transpose to (batch_size, final_seq_len, out_channels[-1])
        x = x.transpose(1, 2)  # swap channel dim and sequence dim

        # --- LSTM ---
        # out shape: (batch_size, final_seq_len, lstm_hidden_size)
        # (hn, cn) have shape: (lstm_num_layers, batch_size, lstm_hidden_size)
        out, (hn, cn) = self.lstm(x)

        # We can take the last timestep's output from 'out'
        # shape: (batch_size, lstm_hidden_size)
        out = out[:, -1, :]

        # --- Classifier ---
        out = self.dropout(out)
        logits = self.fc(out)

        return logits
