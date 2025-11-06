import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F 

class FacialRecognizer:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def process_numpy_array(self, image_array: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor for FaceNet"""
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            return None

        # BGR -> RGB and resize to 160x160
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (160, 160))
        
        # Normalize to [-1, 1] and convert to tensor
        image_normalized = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        
        return image_tensor

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract 512-D embeddings using FaceNet"""
        if image_tensor is None:
            return None
        
        # Add batch dimension if needed
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        with torch.no_grad():
            embeddings = self.model(image_tensor.to(self.device).float())
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def process_and_extract_from_array(self, image_array: np.ndarray) -> torch.Tensor:
        """Process numpy array and extract features"""
        processed_tensor = self.process_numpy_array(image_array)
        return self.extract_features(processed_tensor)

    def process_and_extract(self, image_path: str) -> torch.Tensor:
        """Load image from file and extract features"""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None
        return self.process_and_extract_from_array(img_bgr)

# Simple usage example
if __name__ == "__main__":
    recognizer = FacialRecognizer()
    


