
import numpy as np
from skorch import NeuralNetClassifier
from torch import nn

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


import math

def default_channel_func(C):
    """
    Hacky hardcoding
    """
    if C == 46:
        return 92
    elif C == 92:
        return 184
    else:
        return math.floor(C * 3 / 4)
    

class BOLDCNN(nn.Module):
    def __init__(
        self,
        n_channels,
        window_size,
        num_conv_blocks=2,
        num_fc_blocks=1,
        num_pred_classes=3,
        conv_k=3,
        pool_k=2,
        pool_s=2, # Is it worth tuning?
        channel_func=default_channel_func,
        dropout_rate=0.5
        
    ):

        super().__init__()

        self.C_input = n_channels
        self.window_size = window_size

        self._build_model(
            num_conv_blocks,
            num_fc_blocks,
            num_pred_classes,
            conv_k=conv_k,
            pool_k=pool_k,
            pool_s=pool_s,
            channel_func=channel_func,
            dropout_rate=dropout_rate
        )

    
    def _build_model(
        self,
        num_conv_blocks,
        num_fc_blocks,
        num_pred_classes,
        conv_k,
        pool_k,
        pool_s, # Is it worth tuning?
        channel_func,
        dropout_rate
    ):

        # These layers are applied element wise
        # so no dynamic construction is required
        relu = nn.ReLU()
        dropout = nn.Dropout(p=dropout_rate)

        L = self.window_size
        C = self.C_input

        print(C, L)
        self.conv_layers = []
        self.fc_layers = []
        
        for i in range(num_conv_blocks):
            C_out = channel_func(C)

            conv = nn.Conv1d(
                C,
                out_channels=C_out, # TODO Choose that in a smarter way
                kernel_size=conv_k,
                stride=1,
                padding="same" # Length reduction is done by pooling
            )
            pool = nn.MaxPool1d(kernel_size=pool_k, stride=pool_s)
            bn = nn.BatchNorm1d(C_out)

            self.conv_layers.append(conv)
            self.conv_layers.append(relu)
            self.conv_layers.append(bn)
            self.conv_layers.append(pool)

            setattr(self, f"conv_{i}", conv)
            setattr(self, f"relu_{i}", relu)
            setattr(self, f"bn_{i}", bn)
            setattr(self, f"pool_{i}", pool)

            L = (L - pool_k) // (pool_s) + 1
            C = C_out
            print(C, L)

        if num_fc_blocks < 1:
            raise ValueError(
                "At least one linear layer is required to map to classes"
            )
        num_units = C * L
        for i in range(1, num_fc_blocks):

            fc = nn.Linear(num_units, num_units)
            
            self.fc_layers.append(dropout)
            self.fc_layers.append(fc)
            self.fc_layers.append(relu)

            setattr(self, f"dropout_{i}", dropout)
            setattr(self, f"fc_{i}", fc)
            setattr(self, f"relu_fc_{i}", relu)

        classification_layer = nn.Linear(num_units, num_pred_classes)

        self.classification_dropout = dropout
        self.classification_layer = classification_layer
        
        self.fc_layers.append(dropout)
        self.fc_layers.append(classification_layer)

    def forward(self, x):
        batch_size = x.shape[0]

        #print(x.shape)
        #print("Conv layers : ")
        for layer in self.conv_layers:
            x = layer(x)
            #print(type(layer), x.shape)

        x = x.view((batch_size, -1))
        #print(x.shape)

        #print("fc layers : ")
        for layer in self.fc_layers:
            x = layer(x)
            #print(x.shape)
        
        return x

class WindowNetClassifier(NeuralNetClassifier):
    def sliding_inference(self, segment, stride=4, k=3):
        n_TR = segment.shape[-1] #  # CNN expects (N, C_in, L)
        scores = np.zeros((k, n_TR))
        divisors = np.zeros((n_TR,), dtype=int)
        window_size = self.module__window_size
    
        for wb in range(0, n_TR - window_size, stride):
            inp = segment[..., wb:wb+window_size].reshape(1, -1, window_size)

            pred = self.predict_proba(inp).squeeze()
            broadcast_pred = np.stack([pred for _ in range(window_size)], axis=-1)

            scores[..., wb:wb+window_size] += broadcast_pred

            divisors[wb:wb+window_size] += 1
    
        pred_msk = divisors > 0
        preds = scores[:, pred_msk] / divisors[pred_msk]

        return preds, pred_msk

# TODO Dropout, maxpool
class Simple1DCNN(nn.Module):
    def __init__(self, n_channels, window_size, num_pred_classes=3):
        self.n_channels = n_channels
        self.window_size=window_size
        super(Simple1DCNN, self).__init__()
        
        self.layer1 = nn.Conv1d(
            in_channels=n_channels,
            out_channels=10,
            kernel_size=8,
            stride=1
        )
        self.act1 = nn.ReLU()
        self.layer2 = nn.Conv1d(in_channels=10, out_channels=3, kernel_size=5, stride=1)
        self.fc = nn.Linear(in_features=39, out_features=num_pred_classes) # Formule de fou ou pooling

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = x.view((batch_size, -1))
        logits = self.fc(x)

        return logits

