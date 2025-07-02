# YOLO People Counter Configuration

# Model settings
MODEL_PATH = "models/yolo11s.pt"
DEVICE = "cuda:0"  # or "cpu"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
TARGET_SIZE = (1280, 720)  # (width, height)

# Directory paths
SAMPLES_DIR = "samples"
RESULTS_DIR = "results"
MODELS_DIR = "models"

# Supported image extensions
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

# Display settings
SHOW_IMAGES = True
DISPLAY_DURATION = 1000  # milliseconds
