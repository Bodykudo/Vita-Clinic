import torch
from torchvision import transforms
import numpy as np
import pydicom
from PIL import Image
from .model import ResNet50Network


class MammographyInferenceService:
    """
    Inference service for mammography cancer classification.
    Combines image and metadata inputs for prediction.
    """

    def __init__(self, model_path: str, device=None):
        """
        Initialize the service with the pre-trained model.

        Args:
            model_path (str): Path to the model weights.
            device (torch.device): Device to load the model (CPU or GPU).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_model(self, model_path: str):
        """
        Load the pre-trained model.

        Args:
            model_path (str): Path to the model file.

        Returns:
            ResNet50Network: Loaded model.
        """
        model = ResNet50Network(output_size=2, no_columns=4)  # Binary classification
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def preprocess(self, dicom_path: str):
        """
        Preprocess DICOM image and metadata.

        Args:
            dicom_path (str): Path to the DICOM file.

        Returns:
            dict: Preprocessed image and metadata.
        """
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array.astype(np.float32)

        # Normalize and duplicate grayscale image to RGB
        image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3)
        image = Image.fromarray(image.astype(np.uint8))
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Extract metadata
        metadata = {
            "laterality": 0 if dicom.get("ImageLaterality") == "L" else 1,
            "view": {"CC": 0, "MLO": 1}.get(dicom.get("ViewPosition"), -1),
            "age": int(dicom.get("PatientAge", "058")[:3]),
            "implant": int(dicom.get("ImplantPresent", 0)),
        }
        metadata_tensor = torch.tensor([metadata[key] for key in metadata]).unsqueeze(0).to(self.device)

        return {"image": image, "metadata": metadata_tensor}

    def predict(self, dicom_path: str):
        """
        Perform inference on the given DICOM file.

        Args:
            dicom_path (str): Path to the DICOM file.

        Returns:
            dict: Predicted class and probability.
        """
        preprocessed = self.preprocess(dicom_path)
        with torch.no_grad():
            output = self.model(preprocessed["image"], preprocessed["metadata"])
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, predicted_class = torch.max(probabilities, 1)

        return {
            "class": "cancer" if predicted_class.item() == 1 else "no cancer",
            "probability": top_prob.item(),
        }
