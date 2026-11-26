# Food Freshness Detection Using CNN and MobileNetV2

## Project Overview

This project focuses on detecting food freshness using Deep Learning techniques. It uses Convolutional Neural Networks (CNN) and MobileNetV2 to classify food images based on their freshness.

## Technologies Used

- Python
- TensorFlow / Keras
- CNN
- MobileNetV2
- Streamlit
- NumPy
- OpenCV

## Project Files

- `app_fixed.py` – Main application file.
- `class_names.json` – Contains class labels.
- `food_freshness_cnn.keras` – Trained CNN model.
- `food_freshness_mobilenet.keras` – Trained MobileNetV2 model.

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Food-Freshness-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app_fixed.py
```

## Models

This project uses two deep learning models:

1. CNN
2. MobileNetV2

## Dataset

Dataset Link: https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset

## Future Scope

- Improve model accuracy.
- Support more food categories.
- Deploy the application for real-world use.
