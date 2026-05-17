# Fine-Grained Hazelnut Classification

This repository provides a reproducible pipeline for fine-grained hazelnut cultivar classification using handcrafted features, classical machine learning models, CNN classifiers, and YOLO classification models.

The workflow includes duplicate removal, foreground cropping, image resizing, class-balanced augmentation, train/validation/test splitting, handcrafted feature extraction, model training, independent test evaluation, confusion matrix generation, and interpretability visualizations.

## Repository Purpose

This repository was prepared to support the reproducibility of the manuscript submitted to **The Visual Computer**. It includes the source code, preprocessing scripts, training scripts, test scripts, visualization scripts, and instructions required to reproduce the reported experiments.

## Dataset Organization

Place the original images under `data/raw` using one folder per class:

```text
data/
└── raw/
    ├── Cakildak/
    ├── Damat/
    ├── Devedisi/
    ├── Karafindik/
    ├── Palaz/
    ├── Sivri/
    ├── Tombul/
    └── Yagli/
```

The scripts expect this ImageFolder-style structure. If images are placed directly inside `data/raw` without class subfolders, the duplicate-removal and dataset-processing scripts may not find the images correctly.

## Recommended Project Structure

```text
Fine-Grained-Hazelnut-Classification/
├── data/
│   ├── raw/
│   ├── clean/
│   ├── cropped/
│   ├── resized_640/
│   ├── balanced/
│   └── split/
│       ├── train/
│       ├── val/
│       └── test/
├── outputs/
│   ├── features/
│   ├── ml_results/
│   ├── cnn_results/
│   ├── cnn_test_results/
│   ├── yolo_results/
│   ├── yolo_test_results/
│   └── figures/
├── scripts/
│   ├── 01_remove_duplicates.py
│   ├── 02_crop_foreground.py
│   ├── 03_resize_letterbox_640.py
│   ├── 04_balance_augment_dataset.py
│   ├── 05_split_balanced_dataset.py
│   ├── 06_extract_handcrafted_features.py
│   ├── 07_train_ml_models_with_val_test.py
│   ├── 08_train_cnn_models.py
│   ├── 09_test_cnn_models.py
│   ├── 10_train_yolo_models.py
│   ├── 11_test_yolo_models.py
│   ├── 12_visualize_handcrafted_features.py
│   ├── 13_generate_cnn_attention_maps.py
│   ├── 14_generate_yolo_attention_maps.py
├── src/
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```
Install dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
numpy
pandas
opencv-python
Pillow
scikit-image
scikit-learn
matplotlib
joblib
tqdm
torch
torchvision
timm
ultralytics
rembg
onnxruntime
```

For GPU training, install a PyTorch version compatible with your CUDA version from the official PyTorch installation page.

## Full Reproducible Workflow

### 1. Remove duplicate images

```bash
python scripts/01_remove_duplicates.py --input data/raw --output data/clean
```

This step copies unique images from the raw class folders to `data/clean`.

### 2. Crop foreground hazelnut regions

```bash
python scripts/02_crop_foreground.py --input data/clean --output data/cropped --padding-percent 10 --model u2net
```

This step uses a rembg-based foreground extraction workflow and saves cropped PNG images.

### 3. Resize images with letterbox padding

```bash
python scripts/03_resize_letterbox_640.py --input data/cropped --output data/resized_640 --size 640
```

This step resizes all images to a square canvas while preserving the aspect ratio.

### 4. Create a balanced augmented dataset

```bash
python scripts/04_balance_augment_dataset.py --input data/resized_640 --output data/balanced --target-count 1000 --seed 42
```

For eight classes, this produces 8,000 images in total, with 1,000 images per class.

### 5. Split the dataset into train, validation, and test sets

```bash
python scripts/05_split_balanced_dataset.py --input data/balanced --output data/split --train-per-class 600 --val-per-class 200 --test-per-class 200 --seed 42 --clear-output
```

This creates the following split:

```text
train: 4,800 images
val  : 1,600 images
test : 1,600 images
```

All ML, CNN, and YOLO experiments should use this same split.

## Handcrafted Feature and Classical ML Pipeline

### 6. Extract handcrafted features

Extract features separately for train, validation, and test sets:

```bash
python scripts/06_extract_handcrafted_features.py --input data/split/train --output outputs/features/train_features.csv
python scripts/06_extract_handcrafted_features.py --input data/split/val   --output outputs/features/val_features.csv
python scripts/06_extract_handcrafted_features.py --input data/split/test  --output outputs/features/test_features.csv
```

Optional debug visualizations can be generated with:

```bash
python scripts/06_extract_handcrafted_features.py --input data/split/train --output outputs/features/train_features.csv --debug-dir outputs/features/debug
```

### 7. Train and test classical ML models

```bash
python scripts/07_train_ml_models_with_val_test.py --train outputs/features/train_features.csv --val outputs/features/val_features.csv --test outputs/features/test_features.csv --output outputs/ml_results
```

This script trains the classical ML models using the training set, selects/evaluates using the validation set, and reports final performance on the independent test set.

## CNN Pipeline

### 8. Train CNN models

```bash
python scripts/08_train_cnn_models.py --data data/split --output outputs/cnn_results --img-sizes 224 448 640 --epochs 50 --batch-size 16 --patience 5
```

The default CNN models are:

```text
ResNet50
DenseNet121
EfficientNetB2
ConvNeXt
MobileNetV3
```

### 9. Evaluate CNN models on the independent test set

```bash
python scripts/09_test_cnn_models.py --data data/split --weights outputs/cnn_results/models --output outputs/cnn_test_results --img-sizes 224 448 640
```

This script loads trained CNN weights and reports final test-set metrics, including accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.

## YOLO Classification Pipeline

### 10. Train YOLO classification models

```bash
python scripts/10_train_yolo_models.py --data data/split --output outputs/yolo_results --models yolov8n-cls.pt yolo11n-cls.pt yolo26n-cls.pt --img-sizes 224 448 640 --epochs 100 --batch 8 --patience 5
```

### 11. Evaluate YOLO models on the independent test set

```bash
python scripts/11_test_yolo_models.py --data data/split --weights outputs/yolo_results --output outputs/yolo_test_results --models yolov8n-cls yolo11n-cls yolo26n-cls --img-sizes 224 448 640
```

This script evaluates trained YOLO classifiers using the `test` split.

## Interpretability and Visualization

### 12. Visualize handcrafted features

Dataset mode:

```bash
python scripts/12_visualize_handcrafted_features.py --input data/balanced --output outputs/figures/handcrafted_features --samples-per-class 1
```

Single-image mode:

```bash
python scripts/12_visualize_handcrafted_features.py --image data/balanced/Cakildak/example.png --output outputs/figures/handcrafted_features
```

This script generates publication-ready visual explanations of shape, color, and texture descriptors.

### 13. Generate CNN Grad-CAM attention maps

Default batch mode:

```bash
python scripts/13_generate_cnn_attention_maps.py --data data/balanced --weights outputs/cnn_results/models --output outputs/figures/cnn_attention
```

Single model and resolution:

```bash
python scripts/13_generate_cnn_attention_maps.py --data data/balanced --weights outputs/cnn_results/models/DenseNet121_448_best.pth     --output outputs/figures/cnn_attention --model DenseNet121 --img-size 448
```

### 14. Generate YOLO activation-attention maps

```bash
python scripts/14_generate_yolo_attention_maps.py --data data/balanced --weights outputs/yolo_results --output outputs/figures/yolo_attention --models yolov8n-cls yolo11n-cls yolo26n-cls --img-sizes 224 448 640
```

This script uses forward activations to generate YOLO classification attention maps.

## Outputs

Main outputs are saved under `outputs/`:

```text
outputs/features/                  Handcrafted feature CSV files
outputs/ml_results/                Classical ML models and reports
outputs/cnn_results/               CNN training outputs and weights
outputs/cnn_test_results/          CNN independent test results
outputs/yolo_results/              YOLO training outputs
outputs/yolo_test_results/         YOLO independent test results
outputs/figures/                   Publication-ready visualizations
```

## Citation

If you use this repository, please cite the associated article:

```bibtex
@article{kulcu_hazelnut_visual_computer,
  title   = {Fine-Grained Hazelnut Classification Using Handcrafted Features, CNNs, and YOLO-Based Models},
  author  = {Külcü, Sercan},
  journal = {The Visual Computer},
  year    = {2026},
  note    = {Manuscript submitted / accepted version to be updated}
}
```

