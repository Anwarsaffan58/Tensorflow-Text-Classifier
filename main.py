"""
PROJECT: TensorFlow Text Classifier (Sentiment Analysis)
AUTHOR: Saffan
DESCRIPTION: 
    A Natural Language Processing (NLP) pipeline that classifies text 
    as Positive or Negative using Word Embeddings and Neural Networks.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, datasets, preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF console spam

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'vocab_size': 10000,    # Only keep the top 10,000 most frequent words
    'max_length': 256,      # Cut/Pad reviews to 256 words
    'embedding_dim': 16,    # Size of the vector for each word
    'epochs': 10,
    'batch_size': 512,
    'plot_path': 'outputs/training_history.png',
    'report_path': 'outputs/evaluation_report.txt'
}

def ensure_directories():
    os.makedirs('outputs', exist_ok=True)

# ==========================================
# PART 1: DATA PIPELINE (ETL)
# ==========================================
def load_and_process_data():
    """
    Loads IMDb dataset and preprocesses (tokenization + padding).
    """
    print("[1/4] Loading IMDb Movie Review dataset...")
    
    # Load data (already tokenized as integers)
    (train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(
        num_words=CONFIG['vocab_size']
    )

    # Pad sequences so they are all the same length
    # (Neural Networks require fixed input size)
    train_data = preprocessing.sequence.pad_sequences(
        train_data, maxlen=CONFIG['max_length'], padding='post', truncating='post'
    )
    test_data = preprocessing.sequence.pad_sequences(
        test_data, maxlen=CONFIG['max_length'], padding='post', truncating='post'
    )

    print(f"      Training samples: {len(train_data)}")
    print(f"      Test samples: {len(test_data)}")
    
    # Create validation set (first 10,000 samples)
    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]
    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]
    
    return partial_x_train, partial_y_train, x_val, y_val, test_data, test_labels

# ==========================================
# PART 2: MODEL ARCHITECTURE
# ==========================================
def build_model():
    """
    Constructs the Neural Network with an Embedding Layer.
    Structure: Embedding -> GlobalAveragePooling -> Dense -> Output
    """
    print("[2/4] Building TensorFlow Neural Network...")
    
    model = models.Sequential([
        # Layer 1: Word Embeddings (The "NLP" magic)
        # Converts integer word IDs into dense vectors of fixed size
        layers.Embedding(CONFIG['vocab_size'], CONFIG['embedding_dim']),
        
        # Layer 2: Global Average Pooling
        # Averages the vectors to handle variable length text efficiently
        layers.GlobalAveragePooling1D(),
        
        # Layer 3: Dense (Hidden Layer)
        layers.Dense(16, activation='relu'),
        
        # Layer 4: Output Layer (Binary Classification: 0 or 1)
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

# ==========================================
# PART 3: TRAINING & VISUALIZATION
# ==========================================
def visualize_training(history):
    """Plots Accuracy and Loss over time."""
    print("[4/4] Generating training visualization...")
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, 'bo', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, 'bo', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(CONFIG['plot_path'])
    print(f"      Plot saved to {CONFIG['plot_path']}")

# ==========================================
# MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    print("--- STARTING TENSORFLOW NLP PIPELINE ---\n")
    ensure_directories()
    
    # 1. Load Data
    train_x, train_y, val_x, val_y, test_x, test_y = load_and_process_data()
    
    # 2. Build Model
    model = build_model()
    
    # 3. Train Model
    print("\n[3/4] Training Model (This may take a moment)...")
    history = model.fit(
        train_x, train_y,
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        validation_data=(val_x, val_y),
        verbose=1
    )
    
    # 4. Evaluate on Test Data
    results = model.evaluate(test_x, test_y, verbose=0)
    print(f"\n      FINAL TEST ACCURACY: {results[1]*100:.2f}%")
    
    # Save Report
    with open(CONFIG['report_path'], 'w') as f:
        f.write(f"Final Test Accuracy: {results[1]*100:.2f}%\n")
        f.write(f"Test Loss: {results[0]:.4f}\n")
    
    # 5. Visualize
    visualize_training(history)
    
    print("\n--- PIPELINE COMPLETE ---")