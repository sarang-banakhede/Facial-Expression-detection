import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class ExpressionRecognitionModel(nn.Module):
    def __init__(self, num_classes=6):
        super(ExpressionRecognitionModel, self).__init__()
        
        self.base_model = vgg16(weights=VGG16_Weights.DEFAULT)
        self.base_model.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(25088, 256), 
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
