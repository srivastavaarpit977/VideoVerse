# # DEMO VIDEO--https://www.loom.com/share/9d7dc6a85d4c40c29e1062fa5524b664?sid=b7ed10bf-dcdd-4a57-a992-5c104a5c27e0


# Pepsi and CocaCola Brand Detection Model 

This project provides a pipeline for detecting Pepsi and CocaCola Brand Logos in the video file with their respective timestamps.It consists of:

- `model`- It has `best.pt` which is a file that contains the weights and configuration of the model that achieved the best performance during the training process.
- `Approach document`-It contains the workflow of the model as to which libraries I used,how I trained my model etc.
- `test.py`-This file helps in detecting the logo with their respective timestamps and calculate size and distance from the centre.
- `results1.json`-It contains the timestamps of the brand detection.
- `logo_detection.pdf`-It contains the whole process of training the model,the dataset I chose and how I trained my model.

## Table of Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Usage](#usage)
   - [Running the Python Script](#running-the-python-script)
   - [Using the Jupyter Notebook](#using-the-jupyter-notebook)
4. [Output](#output)

## Requirements

- Python 3.7+
- OpenCV
- Ultralytics YOLO
- Jupyter Notebook (for `logo_detection.pdf`)[converted from .ipynb]
- roboflow

## Setup

1. Clone the repository or download the project files.

2. Install the required Python libraries:

    ```bash
    pip install opencv-python ultralytics jupyter roboflow
    ```

3. Make sure you have the trained YOLO model file (`best.pt`) using the `logo_detection.pdf`.

## Usage

### Running the Python Script

The `test.py` script processes a video file to detect the logo of respective brands and record their timestamps in the file `results1.json`.

1. Open a terminal and navigate to the directory containing `test.py`.

2. Run the script:

    ```bash
    python test.py
    ```

3. The script will read the video file, perform logo detection, and save the timestamps of detected objects in a JSON file named `results1.json`.

### Using the Jupyter Notebook

The `logo_detection.pdf` from `logo_detection.ipynb` notebook provides an interactive interface for custom model training on CocaCola and pepsi logo.

## Output

- The `test.py` script outputs a `results1.json` file containing the timestamps of detected objects in the following format:

    ```json
    {
        "Pepsi_pts": [10.1, 10.2, 10.3, ...],
        "CocaCola_pts": [20.3, 31.8, 40.12, ...]
    }
    ```
---

