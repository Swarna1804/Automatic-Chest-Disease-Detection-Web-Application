🫁 Automatic Chest Disease Detection Using Deep Learning & XAI
📌 Overview
This project presents an AI-powered system for automatic detection of 14 thoracic diseases from chest X-ray images using a fine-tuned ResNet50 deep learning model integrated with Explainable Artificial Intelligence (XAI) through Grad-CAM.

The system is deployed as a Streamlit web application, enabling users to upload chest X-ray images, obtain multi-label disease predictions with probability scores, and visualize heatmaps highlighting clinically relevant regions that influenced the model’s decisions.

🚀 Key Features
Multi-label classification of 14 thoracic diseases

Sigmoid-based probability outputs for each disease

Grad-CAM heatmaps for explainable predictions

Real-time inference via Streamlit web interface

User-friendly and clinically interpretable outputs

🧠 Model Architecture
Backbone: ResNet50 (ImageNet pre-trained)

Fine-tuning: Final fully connected layer modified for 14 disease classes

Activation Function: Sigmoid (multi-label classification)

Loss Function: Binary Cross-Entropy (BCE)

Explainability Module: Grad-CAM applied to the final convolutional layer

🛠️ Tech Stack
Programming Language: Python

Deep Learning Framework: PyTorch

Web Framework: Streamlit

Explainable AI: Grad-CAM

Image Processing & Visualization: NumPy, PIL, Matplotlib
