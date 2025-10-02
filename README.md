# Binary Classification: Turkish Rice Species

Barebones binary classification framework expanding on [**Google's ML Crash Course Implementations.**](https://developers.google.com/machine-learning/crash-course/classification) Classifies Turkish rice grains (Cammeo/Osmancik) using morphological features.

**Dataset:** Cinar & Koklu 2019 ([DOI:10.18201/ijisae.2019355381](https://doi.org/10.18201/ijisae.2019355381))

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train baseline model (3 features, ~90% accuracy)
python train.py --experiment baseline

# Train full model (7 features, ~92% accuracy)
python train.py --experiment full

# Compare both
python train.py --experiment compare
```

---

## Python API

```python
from rice_classifier import RiceClassifier, load_rice_data, split_data

# Load data
df = load_rice_data()
train_df, val_df, test_df = split_data(df)

# Train classifier
clf = RiceClassifier(features=['Area', 'Eccentricity', 'Major_Axis_Length'])
clf.fit(train_df, verbose=1)

# Evaluate
metrics = clf.evaluate(test_df)
print(f"Test accuracy: {metrics['accuracy']:.3f}")

# Predict
predictions = clf.predict(test_df)
classes = clf.predict_classes(test_df)

# Deploy
clf.save('model.pkl')
loaded = RiceClassifier.load('model.pkl')
```

---

## CLI Usage

```bash
# Predefined experiments
python train.py --experiment baseline  # 3 features
python train.py --experiment full      # 7 features

# Custom configuration
python train.py \
  --features Area Eccentricity Perimeter \
  --epochs 100 \
  --learning-rate 0.002 \
  --threshold 0.4 \
  --save model.pkl

# Training options
--verbose {0,1,2}      # 0=silent, 1=progress bar, 2=epoch-by-epoch
--save PATH            # Save trained model
```

---

## API Reference

### `Preprocessor`

Feature preprocessor with fit/transform pattern.

```python
prep = Preprocessor(features=['Area', 'Eccentricity'])
prep.fit(train_df)
X_train, y_train = prep.transform(train_df)
X_test, y_test = prep.transform(test_df, return_labels=False)
```

**Methods:**
- `fit(df)` - Fit normalization on training data
- `transform(df, return_labels=True)` - Transform to normalized features
- `fit_transform(df)` - Fit and transform in one step

### `RiceClassifier`

Binary classifier with sklearn-style interface.

```python
clf = RiceClassifier(
    features=['Area', 'Eccentricity'],
    config=ClassifierConfig(learning_rate=0.001, epochs=60)
)
```

**Methods:**
- `fit(df, labels=None, validation_data=None, verbose=0)` - Train model
- `predict(df)` - Predict probabilities
- `predict_classes(df)` - Predict binary classes
- `evaluate(df, labels=None)` - Compute metrics
- `save(path)` - Serialize to disk
- `load(path)` - Load from disk (classmethod)
- `get_history()` - Get training history

### `ClassifierConfig`

```python
config = ClassifierConfig(
    learning_rate=0.001,
    epochs=60,
    batch_size=100,
    threshold=0.35,
    train_ratio=0.8,
    validation_ratio=0.1,
    random_state=100
)
```

### Utility Functions

- `load_rice_data(url, cache_path)` - Load dataset with caching
- `split_data(df, train_ratio, validation_ratio, random_state)` - Split into train/val/test
- `run_experiment(name, features, config, verbose)` - Complete experiment workflow

---

## Architecture

**Data Pipeline:**
1. Load from URL or cache
2. Split 80/10/10 (train/val/test)
3. Z-score normalize features (fit on train)
4. Encode labels (Cammeo=1, Osmancik=0)

**Model:**
```
Input_1 ─┐
Input_2 ─┼─> Concatenate ─> Dense(1, sigmoid) ─> Output
Input_N ─┘
```

- Loss: Binary crossentropy
- Optimizer: RMSprop
- Metrics: Accuracy, Precision, Recall, AUC

**Available Features:**
`Area`, `Perimeter`, `Major_Axis_Length`, `Minor_Axis_Length`, `Eccentricity`, `Convex_Area`, `Extent`

---

## Reproducing Colab Results

| Model | Features | Threshold | Expected Accuracy |
|-------|----------|-----------|-------------------|
| Baseline | 3 (Eccentricity, Major_Axis, Area) | 0.35 | ~90% |
| Full | All 7 features | 0.5 | ~92% |

```bash
python train.py --experiment baseline  # Expected: ~90% test accuracy
python train.py --experiment full      # Expected: ~92% test accuracy
```

---

## Project Structure

```
google-binary-classification/
├── rice_classifier.py    # Core: Preprocessor, Classifier, utilities
├── train.py              # CLI for experiments
├── requirements.txt      # Dependencies (Keras 3.8, TensorFlow 2.18)
├── README.md            # Documentation
└── .gitignore           # Git ignore rules
```

**5 files total** — minimal complexity, maximal signal.

---

## Key Improvements Over Colab

- **DataFrame-first API**: Natural sklearn-style interface
- **Proper serialization**: Save/load entire pipeline (preprocessor + model)
- **Stateless design**: No data hoarding post-training
- **Generalizable**: Works beyond rice dataset as binary classification framework
- **Production-ready**: Type hints, error handling, validation
- **Dependency-free**: Eliminates google-ml-edu package

---

## Requirements

- Python 3.8+
- TensorFlow 2.18
- Keras 3.8
- NumPy 2.0
- Pandas 2.2

Install: `pip install -r requirements.txt`

---

## Citation

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

CC0 (Public Domain) — matches dataset license.

**Repository:** https://github.com/davidkimai/google-binary-classification
