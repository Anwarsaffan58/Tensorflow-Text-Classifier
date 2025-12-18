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

git clone [https://github.com/Anwarsaffan58/Tensorflow-Text-Classifier.git](https://github.com/Anwarsaffan58/Tensorflow-Text-Classifier.git)
cd Tensorflow-Text-Classifier
pip install -r requirements.txt
python main.py
--- STARTING TENSORFLOW NLP PIPELINE ---

[1/4] Loading IMDb Movie Review dataset...
      Training samples: 25000
      Test samples: 25000

[2/4] Building TensorFlow Neural Network...
      Model: "sequential"
      _________________________________________________________________
       Layer (type)                Output Shape              Param #   
      =================================================================
       embedding (Embedding)       (None, None, 16)          160000    
       global_average_pooling1d    (None, 16)                0         
       dense (Dense)               (None, 16)                272       
       dense_1 (Dense)             (None, 1)                 17        
      =================================================================
      Total params: 160,289

[3/4] Training Model...
      Epoch 1/10: accuracy: 0.5715 - val_accuracy: 0.6057
      Epoch 5/10: accuracy: 0.7607 - val_accuracy: 0.7705
      Epoch 10/10: accuracy: 0.8533 - val_accuracy: 0.8392

      FINAL TEST ACCURACY: 83.31%

[4/4] Generating training visualization...
      Plot saved to outputs/training_history.png

📊 Results & AnalysisThe model achieved a Final Test Accuracy of 83.31%.VisualizationsThe training history plot (saved in outputs/) confirms that the model converges steadily without significant overfitting. The loss curve decreases consistently, indicating that the Embedding Layer successfully learned the semantic features of the dataset.🧠 How It Works (The Logic)Tokenization: Words are converted to integers (e.g., "Great" -> 45) based on frequency.Padding: All reviews are forced to a length of 256 words using pad_sequences.Embedding: The model learns to map words to a 16-dimensional vector space.Classification: A Dense layer with a Sigmoid activation function outputs a probability score ($0 \to 1$), where $>0.5$ is Positive and $<0.5$ is Negative.📝 AuthorSaffan
