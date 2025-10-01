# Binary Classification: Turkish Rice Species

Production implementation of Google ML Crash Course binary classification exercise. Classifies rice grains into two species (Cammeo and Osmancik) using morphological measurements.

**Dataset:** Cinar & Koklu 2019 (CC0 License)  
**Citation:** [DOI:10.18201/ijisae.2019355381](https://doi.org/10.18201/ijisae.2019355381)

---

## Features

- **Minimal complexity, maximal signal:** Single-file core implementation (~380 lines)
- **Plug-and-play API:** Simple `fit/predict/evaluate` interface
- **Reproducible:** Fixed random seeds and versioned dependencies
- **Enterprise-ready:** Type hints, error handling, configuration management
- **No external frameworks:** Eliminates google-ml-edu dependency

---

## Installation

```bash
git clone https://github.com/davidkimai/google-binary-classification.git
cd google-binary-classification
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, TensorFlow 2.18, Keras 3.8

---

## Quick Start

### Python API

```python
from rice_classifier import RiceClassifier, Config

# Create classifier with default settings (3 features)
config = Config()
classifier = RiceClassifier(config)

# Train and evaluate
results = classifier.fit(verbose=1)
print(f"Test accuracy: {results['test_accuracy']:.3f}")
print(f"Test AUC: {results['test_auc']:.3f}")

# Make predictions
predictions = classifier.predict(test_features)
binary_predictions = classifier.predict_classes(test_features)
```

### Command Line

```bash
# Train baseline model (3 features, matches Colab)
python train.py --experiment baseline

# Train full model (7 features)
python train.py --experiment full

# Compare both models
python train.py --experiment compare

# Custom configuration
python train.py --features Area Eccentricity Perimeter \
                --epochs 100 \
                --learning-rate 0.002 \
                --threshold 0.4

# Save trained model
python train.py --experiment full --save rice_model.keras
```

---

## Reproducing Colab Results

The original Colab trains two models:

### Baseline Model (3 features)
- Features: `Eccentricity`, `Major_Axis_Length`, `Area`
- Threshold: 0.35
- Expected test accuracy: ~90%

```bash
python train.py --experiment baseline
```

### Full Model (7 features)
- Features: All 7 morphological measurements
- Threshold: 0.5
- Expected test accuracy: ~92%

```bash
python train.py --experiment full
```

### Side-by-Side Comparison
```bash
python train.py --experiment compare
```

**Expected output:**
```
COMPARISON SUMMARY
================================================================
           experiment           features  train_accuracy  ...
  Baseline (3 features)  Eccentricity...          0.9205  ...
      Full (7 features)  Area, Perime...          0.9231  ...
```

---

## API Reference

### `Config`

Dataclass for hyperparameters and settings.

**Key parameters:**
- `input_features` (List[str]): Feature names to use
- `learning_rate` (float): Learning rate for RMSprop optimizer (default: 0.001)
- `number_epochs` (int): Training epochs (default: 60)
- `batch_size` (int): Batch size (default: 100)
- `classification_threshold` (float): Binary classification threshold (default: 0.35)
- `train_ratio` / `validation_ratio` / `test_ratio`: Data split ratios (default: 0.8/0.1/0.1)

**Available features:**
`Area`, `Perimeter`, `Major_Axis_Length`, `Minor_Axis_Length`, `Eccentricity`, `Convex_Area`, `Extent`

### `RiceClassifier`

Main classifier class with scikit-learn-style API.

**Methods:**

#### `__init__(config: Config)`
Initialize classifier with configuration.

#### `fit(verbose: int = 0) -> Dict[str, float]`
Train model and return metrics for train/validation/test splits.

**Returns:** Dictionary with keys like `train_accuracy`, `validation_auc`, `test_precision`, etc.

#### `predict(features: Dict[str, np.ndarray]) -> np.ndarray`
Predict probabilities for positive class (Cammeo).

**Args:**
- `features`: Dict mapping feature names to numpy arrays

**Returns:** Array of probabilities [0, 1]

#### `predict_classes(features: Dict[str, np.ndarray]) -> np.ndarray`
Predict binary classes using configured threshold.

**Returns:** Binary array (0=Osmancik, 1=Cammeo)

#### `evaluate(features: Dict, labels: np.ndarray) -> Dict[str, float]`
Evaluate model on given data.

**Returns:** Dictionary of metrics (accuracy, precision, recall, AUC)

#### `save(path: str)`
Save model to disk in Keras format.

#### `get_history() -> pd.DataFrame`
Get training history with metrics per epoch.

---

## Architecture

### Data Pipeline
1. **Load:** Download from URL or load from cache
2. **Validate:** Check schema and required columns
3. **Split:** 80/10/10 train/validation/test with shuffling (seed=100)
4. **Normalize:** Z-score normalization fitted on training data
5. **Encode:** Binary labels (Cammeo=1, Osmancik=0)

### Model Architecture
```
Input_1 (Eccentricity)  ─┐
Input_2 (Major_Axis)    ─┼─> Concatenate ─> Dense(1, sigmoid) ─> Output
Input_3 (Area)          ─┘
```

- **Type:** Logistic regression (sigmoid activation)
- **Loss:** Binary crossentropy
- **Optimizer:** RMSprop
- **Metrics:** Accuracy, Precision, Recall, AUC

### Training
- **Epochs:** 60 (default)
- **Batch size:** 100
- **Validation:** Computed on separate validation set
- **Final test:** Held-out test set for final metrics

---

## Customization Examples

### Change feature set
```python
config = Config(input_features=['Area', 'Perimeter', 'Extent'])
classifier = RiceClassifier(config)
results = classifier.fit()
```

### Adjust hyperparameters
```python
config = Config(
    learning_rate=0.002,
    number_epochs=100,
    batch_size=50,
    classification_threshold=0.4
)
```

### Access training history
```python
classifier = RiceClassifier(config)
classifier.fit(verbose=0)

history = classifier.get_history()
print(history[['accuracy', 'auc']])  # Metrics per epoch
```

### Save and load models
```python
# Train and save
classifier.fit()
classifier.save('my_model.keras')

# Load and use (note: requires manual reconstruction)
import keras
loaded_model = keras.models.load_model('my_model.keras')
```

---

## Project Structure

```
google-binary-classification/
├── rice_classifier.py       # Core implementation (data, model, training)
├── train.py                 # CLI entry point for experiments
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

**Total: 5 files** — minimal complexity, maximal signal.

---

## Differences from Original Colab

### Improvements
- **Eliminated dependency** on `google-ml-edu` package
- **Consolidated** scattered notebook cells into cohesive module
- **Added** reproducibility via config management and fixed seeds
- **Implemented** clean API with type hints and docstrings
- **Removed** Colab-specific elements (@title decorators, magic commands)
- **Enhanced** error handling and data validation

### Preserved
- **Identical** model architecture and training procedure
- **Same** hyperparameters and random seeds (42, 100)
- **Matching** metrics and evaluation methodology
- **Expected** ~90-92% test accuracy results

---

## Testing

Manual validation against original Colab:

```python
# Verify baseline model achieves ~90% test accuracy
python train.py --experiment baseline
# Expected: test_accuracy ≈ 0.90

# Verify full model achieves ~92% test accuracy  
python train.py --experiment full
# Expected: test_accuracy ≈ 0.92
```

**Note:** Minor variations (±2%) are expected due to:
- TensorFlow version differences
- Hardware/platform variations
- Floating-point precision

---

## Citation

If you use this implementation, please cite the original dataset:

```bibtex
@article{cinar2019rice,
  title={Classification of Rice Varieties Using Artificial Intelligence Methods},
  author={Cinar, I. and Koklu, M.},
  journal={International Journal of Intelligent Systems and Applications in Engineering},
  volume={7},
  number={3},
  pages={188--194},
  year={2019},
  doi={10.18201/ijisae.2019355381}
}
```

---

## License

This implementation follows the dataset's CC0 license. See [Kaggle dataset page](https://www.kaggle.com/datasets/muratkokludataset/rice-dataset-commeo-and-osmancik) for details.

---

## Contributing

This repository implements Google's ML Crash Course exercise. For bugs or improvements, please open an issue at:

**https://github.com/davidkimai/google-binary-classification**

---

## Contact

**Repository:** https://github.com/davidkimai/google-binary-classification  
**Course:** [Google ML Crash Course - Binary Classification](https://developers.google.com/machine-learning/crash-course)
