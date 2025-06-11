### CIFAR-10 Image Classification with Convolutional Neural Networks

#### Project Overview

This repository implements a convolutional neural network (CNN) to classify images from the CIFAR-10 dataset. The project emphasizes reproducibility, modularity, and rigorous evaluation.

- **Objective:** Construct and evaluate a CNN to categorize 60,000 32×32 color images into 10 semantic classes.  
- **Dataset:** CIFAR-10 (50,000 training images; 10,000 test images) across classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.  
- **Contributions:**  
  1. Data normalization and one-hot label encoding pipeline.  
  2. On-the-fly data augmentation (random flips and translations).  
  3. Three-block CNN architecture with dropout regularization.  
  4. Hyperparameter optimization using Adam and learning-rate scheduling.  
  5. Reproducible Jupyter Notebook workflow.  

#### Architectural Summary

The network follows a sequential design:

Input: 32×32×3 images
→ [Conv2D(32 filters, 3×3) → ReLU
   → Conv2D(32 filters, 3×3) → ReLU
   → MaxPooling(2×2) → Dropout(0.25)] × 3
→ Flatten
→ Dense(512 units) → ReLU → Dropout(0.5)
→ Dense(10 units) → Softmax

	•	Activation: ReLU
	•	Regularization: Dropout (25% in conv blocks; 50% before final dense)
	•	Loss: Categorical Crossentropy
	•	Optimizer: Adam (default learning rate)

#### Installation & Setup

**Clone the repository and install dependencies:**

```bash
git clone https://github.com/ashleshakadam/CNN-Powered-CIFAR-10-Classifier-with-Real-Time-Augmentation.git
cd CNN-Powered-CIFAR-10-Classifier-with-Real-Time-Augmentation
```

**Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

**Install Python packages:**

```bash
pip install -r requirements.txt
```

#### Usage

**Launch Jupyter Notebook and run the analysis pipeline:**

```bash
jupyter notebook
```

**Open CIFAR-10_Image_Classification.ipynb and execute cells sequentially:**
	1.	Data loading and preprocessing
	2.	Model definition
	3.	Training and validation
	4.	Evaluation and visualization

**Results & Evaluation**

The model achieves ≥ 90% test accuracy on the 10,000-image test set. Training and validation curves demonstrate stable convergence with minimal overfitting.

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")

**Extensibility**
	•	Integrate residual architectures (ResNet, DenseNet) for deeper representation.
	•	Automate hyperparameter search via Bayesian optimization.
	•	Implement k-fold cross-validation to assess robustness.

**License**

This project is licensed under the MIT License.  
See [LICENSE](https://github.com/ashleshakadam/CNN-Powered-CIFAR-10-Classifier-with-Real-Time-Augmentation/blob/main/LICENSE) for full terms.

**References**
	1.	Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. University of Toronto Technical Report.
	2.	Chollet, F., et al. (2023). Deep Learning with Python (2nd ed.). Manning Publications.

