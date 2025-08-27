---

# 🔢 SVHN Digit Classification using Random Forest

This project focuses on building a machine learning model using the SVHN (Street View House Numbers) dataset to recognize digits from real-world house number images. The model uses a Random Forest Classifier trained on a subset of the `extra_32x32.mat` data.

---

## 📁 Files

NB: Filesize of extra_32x32.mat is too large to upload here. Available at http://ufldl.stanford.edu/housenumbers/ 

| File                       | Description                                                             |
| -------------------------- | ----------------------------------------------------------------------- |
| `extra_32x32.mat`          | SVHN dataset in MATLAB `.mat` format (images + labels)                  |
| `image_processing.ipynb`   | Jupyter notebook with all preprocessing, training, and evaluation steps |
| `Image README.txt`         | This documentation file                                                 |

---

## 🎯 Objectives

* Load and preprocess image data from `.mat` format
* Train a Random Forest model to classify digits (0–9)
* Evaluate performance using accuracy

---

## 📊 Dataset Overview

* Images: 32×32 RGB digits from real-world scenes (house numbers)
* Labels: Digits 1–10 (where label `10` represents digit `0`)
* File used: `extra_32x32.mat`, a supplement to the main SVHN dataset

---

## ⚙️ Preprocessing Steps

1. Load data using `scipy.io.loadmat()`
2. Transpose image array from shape `(32, 32, 3, N)` to `(N, 32, 32, 3)`
3. Flatten each image into a 1D array of 3,072 features (`32 × 32 × 3`)
4. Subset the dataset to reduce training time (e.g. 10,000 samples)
5. Reshape labels with `.ravel()` to make them 1D

```python
X = X.reshape(-1, 32 * 32 * 3)
y = y.ravel()
```

---

## 🧠 Model: Random Forest

* Implemented using `sklearn.ensemble.RandomForestClassifier`
* Trained with:

  * All CPU cores (`n_jobs = -1`)
  * 80/20 train-test split
  * Reproducibility via `random_state = 50`

```python
clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)
```

---

## 📈 Results

* On a sample of 10,000 images:

```plaintext
Accuracy with 10000 samples: ~0.70 – 0.75 (depending on subset)
```

* This is decent considering:

  * No image normalization
  * No feature extraction (e.g. edge detection or HOG)
  * No deep learning — only Random Forests

---

## 🔍 Observations

* Label `10` corresponds to the digit `0` — you may want to remap:

  ```python
  y[y == 10] = 0
  ```
* Larger sample sizes will improve generalization but increase training time significantly.
* Visualizing predictions helps to debug model output (see `imshow()` block).

---

## 📦 Requirements

* Python 3.x
* Libraries:

  * `numpy`
  * `matplotlib`
  * `scipy`
  * `scikit-learn`
  * `jupyter` (optional, for notebook interface)

Install them using:

```bash
pip install numpy matplotlib scipy scikit-learn jupyter
```

---

## ▶️ How to Run

1. Place `extra_32x32.mat` in the same folder
2. Launch the notebook:

   ```bash
   jupyter notebook svhn_rf_classifier.ipynb
   ```
3. Run all cells to:

   * Load and preprocess data
   * Train the classifier
   * View accuracy and test predictions

---

## ✅ Future Improvements

* Normalize image pixel values to `[0, 1]` range
* Use dimensionality reduction (PCA) to improve training time
* Evaluate using confusion matrix and classification report

---

