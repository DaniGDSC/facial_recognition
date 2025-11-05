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

recognizer = FacialRecognizer()

# 1. Extract embeddings from image file
features = recognizer.process_and_extract("/home/un1/projects/facial_recognition/data/captured_faces/aligned_face_20251105_113243.jpg")
print(f"Features shape: {features.shape}")  # torch.Size([1, 512])
print(f"Features type: {type(features)}")   # <class 'torch.Tensor'>

# 2. Extract from numpy array
img_array = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
features = recognizer.process_and_extract_from_array(img_array)
print(f"Embedding: {features[0][:5]}")  # tensor([ 0.1234, -0.5678,  0.9012, ...])

# 3. Classification (if classifier initialized)
recognizer_with_classifier = FacialRecognizer(num_classes=3)
probs, pred_class, confidence = recognizer_with_classifier.classify(features)
print(f"Probabilities: {probs}")     
print(f"Predicted: {pred_class}")    
print(f"Confidence: {confidence}")   


