[README.md](https://github.com/user-attachments/files/29864398/README.md)
# Multiclass Image Classification — Dog Breed Identification

A deep learning project that classifies images of dogs into **120 breeds** using transfer learning with **InceptionResNetV2** (pretrained on ImageNet) and TensorFlow/Keras.

## Overview

This project builds a convolutional neural network on top of a frozen InceptionResNetV2 backbone to perform multiclass image classification. It is based on the [Kaggle Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification) dataset, which contains labeled images of dogs across 120 different breeds.

## Dataset

- `labels.csv` — image IDs mapped to their breed label
- `train/` — training images
- `test/` — test images for inference

The dataset is expected to be mounted from Google Drive when run in Google Colab (paths like `/content/drive/MyDrive/dog/train`), but can be adapted to run locally by updating the file paths.

## Model Architecture

- **Base model:** InceptionResNetV2 (`include_top=False`, `weights='imagenet'`), frozen during training
- **Head:**
  - Global Average Pooling 2D
  - Batch Normalization
  - Dense(512, relu)
  - Dense(256, relu)
  - Dropout(0.5)
  - Dense(128, relu)
  - Dense(120, softmax) — one output per breed

**Input shape:** 331 × 331 × 3
**Loss:** Categorical Crossentropy
**Optimizer:** Adam
**Metric:** Accuracy

## Training Details

- Train/validation split: 80% / 20%
- Data augmentation: rescaling (1/255) and horizontal flip via `ImageDataGenerator`
- Batch size: 32
- Early stopping on validation loss (patience = 10, min_delta = 0.001, restores best weights)
- Epochs: up to 25 (subject to early stopping)

## Results

The model was trained for 16 epochs (early stopping restored the best weights) and achieved:

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 0.9449 | 0.9023 |
| Loss (Cross Entropy) | 0.1790 | 0.4740 |

**Training and Validation Accuracy**

![Training and Validation Accuracy](images/training_validation_accuracy.png)

**Training and Validation Loss**

![Training and Validation Loss](images/training_validation_loss.png)

Training accuracy climbs steadily and settles around 94–95%, while validation accuracy plateaus near 90%, with validation loss beginning to drift upward after roughly epoch 5 — indicating the model starts mildly overfitting past that point, which is where early stopping helps preserve the best-performing weights.

## Inference

The trained model is saved as `Model.h5` and used to generate predictions on the test set, which are written to `submission.csv` in the format:

```
image_id,label
```

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn (classification report, confusion matrix)
- OpenCV

## Getting Started

1. Clone this repository and open `Multiclass_image_classificationproject.ipynb` in Jupyter or Google Colab.
2. Download the [Dog Breed Identification dataset](https://www.kaggle.com/c/dog-breed-identification) and update the `train_path`, `test_path`, and `labels.csv` paths in the notebook.
3. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas seaborn matplotlib scikit-learn opencv-python
   ```
4. Run all cells to train the model and generate predictions.

## Reference

This project follows the approach outlined in [GeeksforGeeks: Multiclass Image Classification using Transfer Learning](https://www.geeksforgeeks.org/deep-learning/multiclass-image-classification-using-transfer-learning/).

## License

Add a license of your choice (e.g., MIT) here.
