# 📚 TensorFlow Text Classifier (Sentiment Analysis)

An NLP pipeline built with TensorFlow/Keras to classify textual data using Word Embeddings and Neural Networks.

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-Binary_Classification-blue)

## 🚀 Project Overview
Text classification is a foundational task in **Natural Language Processing (NLP)**. This project implements a Deep Learning model to categorize movie reviews from the **IMDb Dataset** as either **Positive** or **Negative**.

Unlike traditional "Bag of Words" approaches, this model utilizes a **Word Embedding Layer** to learn dense vector representations of words. This allows the system to capture semantic relationships (e.g., understanding that "fantastic" and "great" are mathematically similar), resulting in higher accuracy and better generalization.

### 🔑 Key Features
- **End-to-End Pipeline**: Automated data loading, tokenization, sequence padding, and tensor formatting.
- **Word Embeddings**: Transforms sparse integer indices into dense 16-dimensional vectors.
- **Efficient Architecture**: Utilizes `GlobalAveragePooling1D` to reduce dimensionality while preserving semantic features, making the model lightweight and fast.
- **Performance Tracking**: Real-time visualization of Loss and Accuracy metrics during training.

## 🛠️ Tech Stack
- **TensorFlow / Keras**: Core Deep Learning framework.
- **NumPy**: High-performance matrix operations.
- **Matplotlib**: Visualization of learning curves.
- **Scikit-Learn**: Metrics and evaluation.

## 📂 Project Structure
```text
Tensorflow-Text-Classifier/
│
├── outputs/                 # Stores generated plots and logs
│   ├── training_history.png # Loss/Accuracy visualization
│   └── evaluation_report.txt
├── main.py                  # Main execution pipeline
├── requirements.txt         # Dependency list
└── README.md                # Project documentation
