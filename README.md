# Real-Time Emotion Recognition

A Python application that uses OpenCV and a custom-trained Convolutional Neural Network (CNN) with TensorFlow/Keras to detect human faces from a live webcam feed and classify their emotions in real-time.

## Demo

*(This is a great place to add a GIF of your project in action! You can use tools like Giphy Capture or ScreenToGif to record your screen.)*

![Demo GIF](c:\Users\nifem\Downloads\emotiongif.gif)

---

## Features

-   **Real-Time Face Detection:** Utilizes OpenCV's Haar Cascade classifier to locate faces in a video stream.
-   **Multi-Class Emotion Classification:** Classifies detected faces into one of several emotions (e.g., Happy, Sad, Neutral, Angry, etc.).
-   **Deep Learning Model:** Built with a custom Convolutional Neural Network (CNN) in TensorFlow and Keras.
-   **Data Augmentation:** The model was trained using data augmentation techniques to improve its generalization and robustness.

## Tech Stack

-   Python 3.11
-   OpenCV
-   TensorFlow / Keras
-   NumPy

## Setup and Installation

Follow these steps to set up the project on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the setup is complete, run the main application script from the terminal:

```bash
python realtime_emotion.py
```

A window will open showing your webcam feed. The application will draw a rectangle around detected faces and display the predicted emotion. Press the 'q' key to quit.

## Model Training

The CNN model was trained on the FER-2013 dataset from Kaggle. The initial model exhibited severe overfitting, which was mitigated by introducing data augmentation (random rotations, shifts, zooms, and flips) during the training process. This project demonstrates a complete data science workflow, from data cleaning and model building to diagnosing issues and iterating for improvement.

## Acknowledgements

-   Dataset: [FER-2013 (Challenges in Representation Learning)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)