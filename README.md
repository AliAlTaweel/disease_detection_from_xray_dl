# Chest X-Ray Disease Detection 🩺

A deep learning project for classifying chest X-ray images to detect pneumonia using Convolutional Neural Networks (CNNs).

## 🎯 Project Overview

This project demonstrates how artificial intelligence can support medical diagnosis by building a deep learning model that classifies chest X-ray images as either **Normal** or **Pneumonia**. The model aims to assist healthcare professionals in making faster and more accurate diagnoses.

## 📊 Dataset

**Source**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Dataset Characteristics**:
- Total images: 5,863
- Classes: 2 (`NORMAL`, `PNEUMONIA`)
- Split: Training, Validation, and Test sets
- Format: JPEG images of chest X-rays
- Note: Dataset exhibits class imbalance (more pneumonia cases than normal)

## 🚀 Project Workflow

### 1. Data Exploration
- Load and visualize sample X-ray images from both classes
- Analyze class distribution across training, validation, and test sets
- Identify dataset imbalances and characteristics
- Examine image dimensions and quality

### 2. Data Preprocessing
- **Image Resizing**: Standardize all images (e.g., 150×150 or 224×224 pixels)
- **Normalization**: Scale pixel values to [0, 1] range
- **Data Augmentation**: Apply transformations to balance classes and improve generalization
  - Random rotations
  - Horizontal flips
  - Zoom variations
  - Brightness adjustments

### 3. Model Building

**Approach A: Custom CNN Architecture**
- Convolutional layers (Conv2D) for feature extraction
- MaxPooling layers for dimensionality reduction
- Dropout layers to prevent overfitting
- Dense layers for classification
- Output layer with sigmoid activation

**Approach B: Transfer Learning** (Optional Enhancement)
- Pre-trained models: MobileNet, VGG16, or ResNet
- Fine-tuning on chest X-ray dataset
- Faster training with better performance

### 4. Model Training
- Optimizer: Adam or SGD
- Loss function: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall
- Early stopping and model checkpointing
- Training monitoring with callbacks

### 5. Model Evaluation

**Metrics**:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

**Visualizations**:
- Training vs Validation Loss curves
- Training vs Validation Accuracy curves
- Sample predictions with confidence scores
- Misclassified examples analysis

## 📁 Project Structure

```
chest-xray-detection/
│
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── notebooks/
│   └── chest_xray_classification.ipynb
│
├── models/
│   └── saved_model.h5
│
├── results/
│   ├── plots/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

## 🛠️ Technologies & Libraries

- **Python 3.8+**
- **TensorFlow / Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib / Seaborn**: Visualization
- **Scikit-learn**: Metrics and evaluation
- **OpenCV / Pillow**: Image processing

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/chest-xray-detection.git
cd chest-xray-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

## 💻 Usage

```python
# Load and preprocess data
python preprocess.py

# Train the model
python train.py

# Evaluate the model
python evaluate.py

# Make predictions
python predict.py --image path/to/xray.jpg
```

Or run the complete workflow in Jupyter Notebook:
```bash
jupyter notebook notebooks/chest_xray_classification.ipynb
```

## 📈 Results

After implementing Transfer Learning (MobileNetV2) and addressing the validation set imbalance, the model achieved high diagnostic safety standards, particularly in minimizing missed cases.

### Final Metrics (Test Set)
| Metric | Score |
|--------|-------|
| **Accuracy** | **81.7%** |
| **Pneumonia Recall (Sensitivity)** | **98.9%** |
| **Pneumonia Precision** | **77.8%** |
| **False Negatives** | **4** (out of 390 cases) |

### Performance Visualization
The model demonstrated high stability during training after re-splitting the validation data:
- **Recall Priority**: The model successfully prioritized identifying Pneumonia cases, missing only 4 out of 390 actual positive cases.
- **Training Stability**: The Accuracy and Loss curves showed smooth convergence, indicating that the overfitting issue from earlier versions was resolved through Dropout (0.4) and Data Augmentation.

## 🔍 Key Insights

- **Recall is King**: In medical AI, a False Negative (missing pneumonia) is more dangerous than a False Positive. This model achieved **~99% Recall**, making it a highly effective screening tool.
- **The Validation Trap**: The original dataset's validation folder (16 images) was too small for reliable metrics. Creating a 20% validation split from the training directory was the "turning point" for model stability.
- **Transfer Learning Power**: Moving from a shallow custom CNN to a pre-trained **MobileNetV2** architecture allowed the model to leverage complex feature extraction, drastically reducing the False Negative rate from 37% down to 1%.
- **Normalization Matters**: Using the specific rescaling required by MobileNetV2 ($[-1, 1]$ range) significantly improved the convergence speed and final accuracy.

## 🔍 Key Insights

- **Class Imbalance**: The dataset contains more pneumonia cases, requiring careful handling through augmentation and appropriate metrics
- **Medical Context**: High recall is crucial to minimize false negatives (missing pneumonia cases)
- **Transfer Learning**: Pre-trained models can significantly improve performance with limited medical imaging data
- **Ethical Considerations**: AI should assist, not replace, medical professionals in diagnosis

## 🚧 Challenges & Solutions

**Challenge 1**: Imbalanced dataset
- *Solution*: Data augmentation, class weights, and appropriate evaluation metrics

**Challenge 2**: Overfitting on small dataset
- *Solution*: Dropout, regularization, data augmentation, and early stopping

**Challenge 3**: Interpretability in medical context
- *Solution*: Visualization of predictions, confusion matrix analysis, and error analysis

## 🔮 Future Improvements

- [ ] Multi-class classification (Normal, Bacterial Pneumonia, Viral Pneumonia)
- [ ] Implement Grad-CAM for visualization of model attention
- [ ] Ensemble methods combining multiple models
- [ ] Deploy model as a web application
- [ ] Add explainability features for medical professionals
- [ ] Incorporate patient metadata for enhanced predictions

## 📚 References

- [Chest X-Ray Dataset Paper](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5)
- [Deep Learning for Medical Image Analysis](https://www.nature.com/articles/s41551-018-0195-4)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Ali Al-Taweel**
- Email: alihaltaweel89@gmail.com


## 🙏 Acknowledgments

- Dataset provided by [Kermany et al.](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- Inspired by the need for AI-assisted medical diagnosis
- Thanks to the open-source community for tools and resources

---

**⚠️ Disclaimer**: This project is for educational purposes only. The model should not be used for actual medical diagnosis without proper validation and approval from medical professionals and regulatory bodies.