
import numpy as np
from skorch import NeuralNetClassifier
from torch import nn

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

