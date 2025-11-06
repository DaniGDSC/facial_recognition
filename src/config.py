import torch

CAMERA_INDEX = 0
TARGET_BOX_COLOR = (255, 128, 0)
SUCCESS_COLOR = (0, 255, 0)
FAILURE_COLOR = (0, 0, 255)
TARGET_BOX_SIZE_RATIO = 0.4
MIN_FACE_SIZE_RATIO = 0.6
DESIRED_FACE_SIZE = 160
DESIRED_LEFT_EYE = (0.35, 0.35)
CONFIDENT_THRESHOLD = 0.9

# Liveness detection thresholds
REFLECTION_THRESHOLD = 0.8
EYE_THRESHOLD = 0.8
MIN_CONFIDENCE = 0.8

# Device configuration
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# File Security Settings
MAX_FILE_SIZE_MB = 5  # Maximum file size in megabytes
JPEG_QUALITY = 90  # JPEG compression quality (1-100)

DATABASE_FILE = "/home/un0/projects/facial_recognition/src/database/metadata.db"