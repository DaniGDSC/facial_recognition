import torch
import time
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from facial_detection import FacialDetectorMTCNN
import torch.nn.functional as F 

class FacialRecognizer:
    def __init__(self, device: str = 'cpu', num_classes: int = None):
        self.device = device
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.detector = FacialDetectorMTCNN()
        
        # Optional classifier for identification
        self.classifier = None
        if num_classes is not None:
            self.classifier = torch.nn.Linear(512, num_classes).to(device)
        
        print(f"✓ Facial Recognizer initialized on {device}")

    def process_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image from file path"""
        try:
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"ERROR: Failed to load image from {image_path}")
                return None 
            return self.process_numpy_array(img_bgr)
        except Exception as e: 
            print(f"ERROR: Image processing failed for {image_path}: {e}")
            return None

    def process_numpy_array(self, image_array: np.ndarray) -> torch.Tensor:
        try:
            if image_array is None:
                return None

            # Convert BGR -> RGB (if grayscale, expand to 3 channels)
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                print("ERROR: Expected 3-channel image.")
                return None

            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # Ensure FaceNet expected size 160x160
            if image_rgb.shape[:2] != (160, 160):
                image_rgb = cv2.resize(image_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)

            image_normalized = (image_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
            # To tensor: (H, W, C) -> (C, H, W)
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).contiguous()
            return image_tensor
        
        except Exception as e:
            print(f"ERROR: Array processing failed: {e}")
            return None

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract 512-D embeddings using FaceNet"""
        if image_tensor is None:
            return None
        
        start_time = time.time()

        try:
            if isinstance(image_tensor, np.ndarray):
                image_tensor = torch.from_numpy(image_tensor)
            if isinstance(image_tensor, list):
                image_tensor = torch.stack(
                    [t if isinstance(t, torch.Tensor) else torch.from_numpy(t) for t in image_tensor],
                    dim=0
                )
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            if image_tensor.shape[-2:] != (160, 160):
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor.float(), size=(160, 160), mode='bilinear', align_corners=False
                )
            
            with torch.no_grad():
                embeddings = self.model(image_tensor.to(self.device).float())
                embeddings = F.normalize(embeddings, p=2, dim=1)

            elapsed = time.time() - start_time
            print(f'✓ Generated {embeddings.shape[0]} embedding(s) in {elapsed:.4f}s')
            return embeddings
            
        except Exception as e: 
            print(f"ERROR: Feature extraction failed: {e}")
            return None
    
    def process_and_extract(self, image_path: str) -> torch.Tensor:
        processed_image = self.process_image(image_path)
        if processed_image is None:
            return None

        features = self.extract_features(processed_image)
        return features
    
    def process_and_extract_from_array(self, image_array: np.ndarray) -> torch.Tensor:
        """Process numpy array and extract features"""
        processed_tensor = self.process_numpy_array(image_array)
        if processed_tensor is None:
            return None
        
        features = self.extract_features(processed_tensor)
        return features

    def classify(self, features: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if features is None or self.classifier is None:
            print("ERROR: No features or classifier not initialized")
            return None, None, None
        
        try:
            with torch.no_grad():
                logits = self.classifier(features)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1).values
                
            return probabilities, predicted_class, confidence

        except Exception as e:
            print(f"ERROR: Classification failed: {e}")
            return None, None, None   
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        try:
            if emb1 is None or emb2 is None:
                return 0.0
            if emb1.ndim == 2:
                emb1 = emb1.squeeze(0)
            if emb2.ndim == 2:
                emb2 = emb2.squeeze(0)
            emb1 = F.normalize(emb1, p=2, dim=0)
            emb2 = F.normalize(emb2, p=2, dim=0)
            return torch.dot(emb1, emb2).item()
        except Exception as e:
            print(f"ERROR: Similarity computation failed: {e}")
            return 0.0

    def embed_aligned_bgr(self, aligned_bgr: np.ndarray) -> np.ndarray:
        t = self.process_numpy_array(aligned_bgr)
        emb = self.extract_features(t)
        return emb.squeeze(0).detach().cpu().numpy() if emb is not None else None

def main():   
    # Initialize recognizer
    recognizer = FacialRecognizer(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_classes=None  # Set to number of people if classifier is trained
    )
    # Define path to captured faces folder
    image_path = '/home/un1/projects/facial_recognition/data/captured_faces/aligned_face_20251104_165523.jpg'
    # Process and extract features from an example image
    features = recognizer.process_and_extract(image_path)
    if features is not None:
        print(f"Extracted features shape: {features.shape}")
        # Classify the extracted features
        probabilities, predicted_class, confidence = recognizer.classify(features)
        if predicted_class is not None:
            print(f"Predicted class: {predicted_class.item()}, Confidence: {confidence.item()}")

if __name__ == "__main__":
    main()