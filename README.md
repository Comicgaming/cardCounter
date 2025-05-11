# CardCounter Project

## Overview
The CardCounter project uses YOLOv5 for object detection to identify playing cards. It is designed to train, evaluate, and perform inference on a dataset of playing cards.

## Prerequisites
- Python 3.x installed on your system.
- `pip` (Python package manager) installed.
- Git installed and added to your system PATH.

## Dataset
The dataset used for this project is the Playing Cards dataset, which is a collection of synthetically generated cards blended into various backgrounds. For more details, refer to the dataset's README file located at:
``PlayingCards/README.dataset.txt``

## Setup Instructions
1. Clone or download the project to your computer.
2. Change to the correct directory:
   ```bash
   cd "...\CardCounter"
   ```
3. Install the required dependencies:
   ```bash
   pip install -r yolov5/requirements.txt
   pip install tqdm
   pip install numpy
   pip install opencv-python
   ```

## Running the Script
1. Ensure the dataset is correctly structured in the `PlayingCards` folder.
2. Run the training, evaluation, and inference script:
   ```bash
   python train_model.py
   ```

## Project Structure
```
CardCounter/
├── PlayingCards/
│   ├── data.yaml
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── README.dataset.txt
├── yolov5/
│   ├── train.py
│   ├── val.py
│   ├── detect.py
│   ├── requirements.txt
├── train_model.py
├── README.md
```

## Notes
- The script will automatically clone the YOLOv5 repository and set up the environment if not already done.
- Ensure you have write permissions for the project directory.
- If you encounter issues, verify that all paths in `train_model.py` are correct.

## References
- [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
- [Playing Cards Dataset](https://universe.roboflow.com/object-detection/playing-cards-ow27d)

## About
This project is inspired by the Playing Cards dataset and aims to demonstrate object detection capabilities using YOLOv5. It can be extended for applications like card counting in games such as Blackjack or Poker.
