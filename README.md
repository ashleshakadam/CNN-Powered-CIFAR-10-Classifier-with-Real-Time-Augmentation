CIFAR-10 Image Classification with Convolutional Neural Networks

Project Overview

This repository presents a rigorous implementation of a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset. Through methodical experimentation and disciplined engineering, this project achieves state-of-the-art accuracy while maintaining transparency and reproducibility.
	•	Objective: Develop and evaluate a robust CNN to classify 60,000 32×32 color images into 10 semantic categories.
	•	Dataset Details: CIFAR-10 comprises 50,000 training and 10,000 test images across classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
	•	Key Contributions:
	1.	Comprehensive data normalization and one-hot encoding pipeline.
	2.	On-the-fly data augmentation strategy (random flips, translations) to enhance generalization.
	3.	Custom three-block CNN architecture with dropout regularization.
	4.	Empirical hyperparameter tuning using Adam optimizer and learning-rate scheduling.
	5.	Modular, reproducible Jupyter Notebook workflow.

Architectural Summary

Input (32×32×3)
→ [Conv(32,3×3) → ReLU → Conv(32,3×3) → ReLU → MaxPool(2×2) → Dropout(0.25)] × 3
→ Flatten
→ Dense(512) → ReLU → Dropout(0.5)
→ Dense(10) → Softmax

	•	Activation Functions: ReLU
	•	Regularization: Dropout at 25% in convolutional blocks and 50% before final dense layer
	•	Loss & Optimization: Categorical Crossentropy; Adam optimizer with default learning rate

Installation & Setup
	1.	Clone the repository

git clone https://github.com/ashleshakadam/CNN-Powered-CIFAR-10-Classifier-with-Real-Time-Augmentation.git cifar10-cnn
cd cifar10-cnn

2. **(Optional) Create a virtual environment**
   ```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows

	3.	Install dependencies

pip install -r requirements.txt

## Usage
1. **Launch Jupyter Notebook**
   ```bash
jupyter notebook

	2.	Open CIFAR-10_Image_Classification.ipynb and execute cells sequentially:
	•	Data loading & preprocessing
	•	Model definition
	•	Training & validation
	•	Evaluation & visualization

Results & Evaluation
	•	Test Accuracy: ≥ 90% on the 10,000-image test set
	•	Training Dynamics: Loss and accuracy curves demonstrate stable convergence with minimal overfitting.


**Example evaluation snippet**
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

Extensibility
	•	Integrate deeper architectures (e.g., ResNet, DenseNet)
	•	Automate hyperparameter search via Bayesian optimization
	•	Implement k-fold cross-validation for performance robustness

License

This project is distributed under the MIT License. See LICENSE for details.

References
	1.	Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto.
	2.	Chollet, F. et al. (2023). Deep Learning with Python, 2nd Edition. Manning Publications.
