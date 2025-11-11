# Hand Gesture Recognition & Control System

This project implements a **Hand Gesture Recognition System** that detects real-time hand gestures using **MediaPipe** and controls system functions using **PyAutoGUI** and **keyboard** automation. It leverages **TensorFlow** for gesture classification and provides a static recognition pipeline (non-real-time training mode) to map gestures to specific system commands.

---

## Project Overview

This notebook builds a machine learning pipeline that:
1. **Captures hand landmarks** using MediaPipe’s `Hands` module.  
2. **Processes the coordinates** into feature vectors suitable for model training.  
3. **Trains a TensorFlow neural network** on labeled gesture data.  
4. **Classifies gestures** (e.g., thumbs-up, palm, fist, OK, etc.) using the trained model.  
5. **Maps recognized gestures to actions** such as controlling windows, simulating keypresses, or automating UI interactions.

---

## Features

- Real-time hand detection using **MediaPipe**.  
- Neural network classifier using **TensorFlow / Keras**.  
- System automation via **PyAutoGUI** and **keyboard**.  
- Supports training, saving, and loading models for gesture recognition.  
- Modular design for easy extension with new gestures or actions.

---

## Tech Stack

| Category | Libraries Used |
|-----------|----------------|
| **Computer Vision** | OpenCV, MediaPipe |
| **Deep Learning** | TensorFlow, Keras, Scikit-Learn |
| **Automation** | PyAutoGUI, keyboard, pygetwindow |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | OpenCV windows and frame overlays |

---

## Project Structure

- handgesturesrecog_static.ipynb # Main notebook containing code 

- static_gestures/ # Folder to store gesture data 

- static_gesture_model.h5/ # Saved trained model 

- README.md  # This file 


---

## Clone the Repository
First, download the project to your local machine:
``` bash
git clone https://github.com/snehilojha/GestureControlSystem.git
cd GestureControlSystem
```

## Installation

- Install dependencies:

```bash
pip install keyboard pyautogui pystray pillow opencv-python mediapipe scikit-learn tensorflow pandas numpy
```
If you’re using Anaconda, you can create a virtual environment:
``` bash
conda create -n GestureControlSystem python=3.10
conda activate GestureControlSystem
```

### Open the Notebook

Open the file in Jupyter Notebook or VS Code:
``` bash
jupyter notebook handgesturesrecog_static.ipynb
```
or open directly in VS Code and select the Python kernel.

---

## Enable Camera Access

- Since the system relies on webcam input for detecting hand gestures:

- Make sure your camera is connected and working.

- Allow camera access when prompted.

- Position your hand about 30–60 cm away from the camera in good lighting.

## Run All Cells Sequentially

- Run the imports and initialization cells first.

- If data collection is part of your workflow: (It is recommended to record your own gestures and train them)

- Run the data recording section and perform each gesture multiple times.

- Save your gesture data before training.

- Run the training cell to build and save the model.

- Finally, execute the gesture recognition section to start real-time prediction and automation.

---

## Customize Gesture Actions

In the automation section (look for code using pyautogui or keyboard),
you can modify the mappings to fit your preferences:
``` bash
if action == "play_pause":
        keyboard.send("play/pause media")
    elif action == "next_track":
        keyboard.send("next track")
    elif action == "previous_track":
        keyboard.send("previous track")
```
Simply replace or add more gestures and actions.

---

## Save or Load Trained Model

-You can save your trained model to reuse later:
``` bash
model.save('saved_model/gesture_model.h5')
```

---

## End the Program

To safely exit:

Press q or Esc in the OpenCV window to terminate.

Or stop execution manually in Jupyter.

## How It Works

- **Data Collection:** -

The script records hand landmarks from webcam input using MediaPipe.

Each gesture is labeled manually and stored as a NumPy array.

- **Feature Extraction & Model Training:** -

Landmark coordinates are normalized and flattened into feature vectors.

The data is split using train_test_split and fed into a Sequential neural network (Dense + Dropout layers).

The model learns to classify gesture labels.

- **Gesture Recognition & Automation:** -

Once trained, the model predicts gestures in real time.

Detected gestures trigger mapped system actions (e.g., volume up, minimize window, open tab, etc.) using PyAutoGUI.

---

## Example Actions

| Gesture | Action |
|-----------|----------------|
| **Palm** | Open Youtube Music |
| **Fist** | Play and Pause |
| **Right (Pointing right)** | Next track |
| **Left (Pointing Left)** | Previous track |

- Gestures can be customised in the *music control* section

---

## Performance Evaluation

- Training and test accuracy metrics are logged after model fitting.

- Confusion matrices and gesture-wise performance can be visualized in later stages.

- Model weights and encoders can be saved for reuse.
