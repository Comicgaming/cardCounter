import os
import sys
import time  # For simulating progress delays
import glob  # For dynamic file path handling

# Define constants
yolov5_dir = os.path.join(os.getcwd(), 'yolov5')  # YOLOv5 directory
dataset_path = os.path.join(os.getcwd(), 'PlayingCards/data.yaml')  # Dataset path
test_images_path = os.path.join(os.getcwd(), 'PlayingCards/test/images')  # Test image directory
output_dir = os.path.join(os.getcwd(), 'runs')  # Output directory
epochs = 50  # Number of epochs to train


# Run a system command with optional description
def run_command(command, description=None):
    """Run a system command with error handling and optional description."""
    if description:
        print(description)
    try:
        # Note: Wrapping command itself in double quotes for safety with spaces
        ret_code = os.system(command)
        if ret_code != 0:
            raise RuntimeError(f"Command failed with return code {ret_code}: {command}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


# Validate each important path and dependencies
if not os.path.exists(dataset_path):
    print(f"Error: Dataset file not found at {dataset_path}.")
    sys.exit(1)

if not os.path.exists(test_images_path):
    print(f"Error: Test images directory not found at {test_images_path}.")
    sys.exit(1)

# Check for correct Python version
if sys.version_info < (3, 7):
    print("Error: Python 3.7 or higher is required.")
    sys.exit(1)

# Ensure PyTorch is installed
try:
    import torch

    print("PyTorch is installed.")
except ImportError:
    print("Error: PyTorch not found. Install it before running this script.")
    sys.exit(1)

# Ensure YOLOv5 is installed and up to date
if os.path.exists(yolov5_dir) and os.path.exists(os.path.join(yolov5_dir, '.git')):
    run_command(f'git -C "{yolov5_dir}" pull', "Updating YOLOv5 repository...")  # Pull latest YOLOv5 repo updates
else:
    run_command(f'git clone https://github.com/ultralytics/yolov5.git "{yolov5_dir}"', "Cloning YOLOv5 repository...")

# Install YOLOv5 dependencies
requirements_path = os.path.join(yolov5_dir, "requirements.txt")
if os.path.exists(requirements_path):
    run_command(f'pip install -r "{requirements_path}"', "Installing YOLOv5 dependencies...")
else:
    print(f"Error: {requirements_path} not found. Ensure YOLOv5 is set up correctly.")
    sys.exit(1)

# Ensure YOLOv5 scripts are available
required_files = ['train.py', 'val.py', 'detect.py']
for file in required_files:
    if not os.path.exists(os.path.join(yolov5_dir, file)):
        print(f"Error: {file} not found in {yolov5_dir}. Ensure YOLOv5 repository is cloned correctly.")
        sys.exit(1)

# Ensure output directory exists for saving runs
os.makedirs(output_dir, exist_ok=True)

# ----- TRAINING -----
print("Starting YOLOv5 training...")
run_command(
    f'python "{os.path.join(yolov5_dir, "train.py")}" --img 640 --batch 16 --epochs {epochs} '
    f'--data "{dataset_path}" --weights yolov5s.pt --project "{output_dir}"',
    "Training YOLOv5 model..."
)
print("Training completed.")

# ----- FIND LATEST WEIGHTS -----
# Select the most recent 'best.pt' weights file dynamically
weight_files = glob.glob(os.path.join(output_dir, "train", "exp*", "weights", "best.pt"))
if weight_files:
    latest_weights = max(weight_files, key=os.path.getctime)  # Get the most recently modified weights
else:
    print("Error: No weights file found after training.")
    sys.exit(1)

# ----- EVALUATION -----
print("Evaluating the trained YOLOv5 model...")
run_command(
    f'python "{os.path.join(yolov5_dir, "val.py")}" --data "{dataset_path}" '
    f'--weights "{latest_weights}"',
    "Evaluating the model..."
)
print("Evaluation completed.")

# ----- INFERENCE -----
print("Running inference on test images...")
run_command(
    f'python "{os.path.join(yolov5_dir, "detect.py")}" --source "{test_images_path}" '
    f'--weights "{latest_weights}"',
    "Running inference..."
)
print("Inference completed.")