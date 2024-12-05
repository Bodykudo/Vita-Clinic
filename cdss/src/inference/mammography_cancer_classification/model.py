import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Network(nn.Module):
    """
    A ResNet50-based model for mammography cancer classification.
    Combines image features with metadata for classification.
    """
    def __init__(self, output_size: int, no_columns: int):
        super().__init__()
        self.no_columns = no_columns
        self.output_size = output_size
        self.features = resnet50(pretrained=True)  # Outputs 1000 neurons

        # Metadata processor (CSV features)
        self.csv = nn.Sequential(
            nn.Linear(self.no_columns, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.classification = nn.Linear(1000 + 500, self.output_size)

    def forward(self, image, meta, prints=False):
        """
        Forward pass for the ResNet50Network.

        Args:
            image (torch.Tensor): Preprocessed image tensor.
            meta (torch.Tensor): Metadata tensor.
            prints (bool): Whether to print debug shapes.

        Returns:
            torch.Tensor: Prediction logits.
        """
        if prints:
            print(f"Input Image shape: {image.shape}")
            print(f"Input Metadata shape: {meta.shape}")

        image = self.features(image)
        if prints:
            print(f"Features Image shape: {image.shape}")

        # Metadata processing
        meta = self.csv(meta)
        if prints:
            print(f"Meta Data shape: {meta.shape}")

        combined_features = torch.cat((image, meta), dim=1)
        if prints:
            print(f"Concatenated Data shape: {combined_features.shape}")

        out = self.classification(combined_features)
        if prints:
            print(f"Out shape: {out.shape}")

        return out
