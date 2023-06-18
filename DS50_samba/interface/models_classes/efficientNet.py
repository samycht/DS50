import torch
from torch import nn
import torchvision as tv

class EfficientNetDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = tv.models.efficientnet_v2_s()
        self.nn_detection = torch.nn.Sequential(
            torch.nn.Linear(1000,4)
        )

    def forward(self, x):
        x = self.model(x)
        logits = self.nn_detection(x)
        return logits
    
    def predict(self,logits):
        preds = self.forward(logits)
        preds = torch.sigmoid(preds)
        return preds



