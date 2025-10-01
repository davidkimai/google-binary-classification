"""
Rice Binary Classification
==========================

Production binary classification framework implementing Google ML Crash Course exercise.
Provides generalizable primitives for tabular binary classification with Keras.

Dataset: Cinar & Koklu 2019 (CC0 License)
Citation: https://doi.org/10.18201/ijisae.2019355381

Public API:
    Preprocessor - Feature selection, normalization, transformation
    RiceClassifier - Binary classifier with sklearn-style fit/predict interface
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

import keras
import numpy as np
import pandas as pd


# Set random seeds for reproducibility
keras.utils.set_random_seed(42)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ClassifierConfig:
    """Hyperparameters for binary classification."""
    
    # Training hyperparameters
    learning_rate: float = 0.001
    epochs: int = 60
    batch_size: int = 100
    
    # Classification
    threshold: float = 0.35
    
    # Data split ratios
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    
    # Reproducibility
    random_state: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.learning_rate < 1, "Learning rate must be in (0, 1)"
        assert self.epochs > 0, "Epochs must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 < self.threshold < 1, "Threshold must be in (0, 1)"


# ============================================================================
# Data Loading
# ============================================================================

DATASET_URL = "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
CACHE_PATH = ".rice_dataset_cache.csv"

ALL_FEATURES = [
    'Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
    'Eccentricity', 'Convex_Area', 'Extent'
]


def load_rice_data(url: str = DATASET_URL, cache_path: str = CACHE_PATH) -> pd.DataFrame:
    """
    Load rice dataset from URL or cache.
    
    Args:
        url: Dataset URL
        cache_path: Local cache path
        
    Returns:
        DataFrame with features and 'Class' column
    """
    cache = Path(cache_path)
    
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        try:
            df = pd.read_csv(url)
            df.to_csv(cache, index=False)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {url}: {e}")
    
    # Validate schema
    required = ALL_FEATURES + ['Class']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    
    return df[required]


# ============================================================================
# Preprocessor
# ============================================================================

class Preprocessor:
    """
    Feature preprocessor with fit/transform pattern.
    
    Handles feature selection, Z-score normalization, and label encoding.
    Serializable for deployment pipelines.
    
    Example:
        >>> prep = Preprocessor(features=['Area', 'Eccentricity'])
        >>> prep.fit(train_df)
        >>> X_train, y_train = prep.transform(train_df)
        >>> X_test, y_test = prep.transform(test_df)
    """
    
    def __init__(self, features: List[str]):
        """
        Initialize preprocessor.
        
        Args:
            features: List of feature column names to use
        """
        self.features = features
        self.feature_mean: Optional[pd.Series] = None
        self.feature_std: Optional[pd.Series] = None
        self._is_fitted = False
        
        # Validate features
        invalid = set(features) - set(ALL_FEATURES)
        if invalid:
            raise ValueError(f"Invalid features: {invalid}. Must be subset of {ALL_FEATURES}")
    
    def fit(self, df: pd.DataFrame) -> 'Preprocessor':
        """
        Fit normalization statistics on training data.
        
        Args:
            df: Training dataframe with feature columns
            
        Returns:
            Self for method chaining
        """
        self._validate_dataframe(df, require_class=False)
        
        # Compute normalization statistics
        self.feature_mean = df[self.features].mean()
        self.feature_std = df[self.features].std()
        self._is_fitted = True
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        return_labels: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Transform dataframe to normalized features and encoded labels.
        
        Args:
            df: Input dataframe
            return_labels: Whether to return labels (requires 'Class' column)
            
        Returns:
            (features_dict, labels) where features_dict maps feature names to arrays
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        self._validate_dataframe(df, require_class=return_labels)
        
        # Normalize features
        normalized = (df[self.features] - self.feature_mean) / self.feature_std
        
        # Convert to dict format for Keras multi-input
        features_dict = {
            feature: normalized[feature].to_numpy()
            for feature in self.features
        }
        
        # Encode labels if present
        labels = None
        if return_labels:
            labels = (df['Class'] == 'Cammeo').astype(int).to_numpy()
        
        return features_dict, labels
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        return_labels: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """Fit and transform in one step."""
        return self.fit(df).transform(df, return_labels=return_labels)
    
    def _validate_dataframe(self, df: pd.DataFrame, require_class: bool = True):
        """Validate dataframe has required columns."""
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing features: {missing}")
        
        if require_class and 'Class' not in df.columns:
            raise ValueError("DataFrame missing 'Class' column")


# ============================================================================
# Model Architecture
# ============================================================================

def build_model(
    features: List[str],
    learning_rate: float,
    threshold: float
) -> keras.Model:
    """
    Build binary classification model.
    
    Architecture: Multi-input -> Concatenate -> Dense(1, sigmoid)
    
    Args:
        features: Feature names
        learning_rate: Optimizer learning rate
        threshold: Classification threshold for metrics
        
    Returns:
        Compiled Keras model
    """
    # Create input for each feature
    inputs = [keras.Input(name=feat, shape=(1,)) for feat in features]
    
    # Concatenate and classify
    x = keras.layers.Concatenate()(inputs)
    output = keras.layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    # Compile with metrics
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy', threshold=threshold),
            keras.metrics.Precision(name='precision', thresholds=threshold),
            keras.metrics.Recall(name='recall', thresholds=threshold),
            keras.metrics.AUC(num_thresholds=100, name='auc'),
        ]
    )
    
    return model


# ============================================================================
# Classifier
# ============================================================================

class RiceClassifier:
    """
    Binary classifier for tabular data with sklearn-style API.
    
    Example:
        >>> # Train on rice dataset
        >>> clf = RiceClassifier(features=['Area', 'Eccentricity', 'Major_Axis_Length'])
        >>> clf.fit(train_df, train_labels)
        >>> 
        >>> # Predict on new data
        >>> predictions = clf.predict(test_df)
        >>> 
        >>> # Save for deployment
        >>> clf.save('model.pkl')
        >>> loaded = RiceClassifier.load('model.pkl')
    """
    
    def __init__(
        self,
        features: List[str],
        config: Optional[ClassifierConfig] = None
    ):
        """
        Initialize classifier.
        
        Args:
            features: List of feature names to use
            config: Hyperparameter configuration (uses defaults if None)
        """
        self.features = features
        self.config = config or ClassifierConfig()
        self.preprocessor = Preprocessor(features)
        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict] = None
    
    def fit(
        self,
        df: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        validation_data: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
        verbose: int = 0
    ) -> 'RiceClassifier':
        """
        Train classifier on data.
        
        Args:
            df: Training dataframe with features and optionally 'Class' column
            labels: Training labels (optional if df has 'Class' column)
            validation_data: Optional (val_df, val_labels) tuple for validation
            verbose: Training verbosity (0=silent, 1=progress, 2=epoch)
            
        Returns:
            Self for method chaining
        """
        # Prepare training data
        X_train, y_from_df = self.preprocessor.fit_transform(
            df,
            return_labels=('Class' in df.columns)
        )
        y_train = labels if labels is not None else y_from_df
        
        if y_train is None:
            raise ValueError("Must provide labels or include 'Class' column in dataframe")
        
        # Prepare validation data if provided
        val_data = None
        if validation_data is not None:
            val_df, val_labels = validation_data
            X_val, y_val_from_df = self.preprocessor.transform(
                val_df,
                return_labels=('Class' in val_df.columns)
            )
            y_val = val_labels if val_labels is not None else y_val_from_df
            val_data = (X_val, y_val)
        
        # Build model
        self.model = build_model(
            self.features,
            self.config.learning_rate,
            self.config.threshold
        )
        
        # Train
        history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=val_data,
            verbose=verbose
        )
        
        self.history = history.history
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            df: Dataframe with feature columns
            
        Returns:
            Array of probabilities for positive class (Cammeo)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X, _ = self.preprocessor.transform(df, return_labels=False)
        return self.model.predict(X, verbose=0).flatten()
    
    def predict_classes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict binary classes.
        
        Args:
            df: Dataframe with feature columns
            
        Returns:
            Binary array (0=Osmancik, 1=Cammeo)
        """
        probs = self.predict(df)
        return (probs >= self.config.threshold).astype(int)
    
    def evaluate(self, df: pd.DataFrame, labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            df: Evaluation dataframe
            labels: True labels (optional if df has 'Class' column)
            
        Returns:
            Dictionary of metric name to value
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X, y_from_df = self.preprocessor.transform(
            df,
            return_labels=('Class' in df.columns)
        )
        y = labels if labels is not None else y_from_df
        
        if y is None:
            raise ValueError("Must provide labels or include 'Class' column in dataframe")
        
        results = self.model.evaluate(X, y, verbose=0, return_dict=True)
        return {k: v for k, v in results.items() if k != 'loss'}
    
    def save(self, path: str):
        """
        Save classifier to disk (preprocessor + model).
        
        Args:
            path: File path (.pkl extension recommended)
        """
        if self.model is None:
            raise ValueError("No model to save. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor and config with pickle
        state = {
            'features': self.features,
            'config': self.config,
            'preprocessor': self.preprocessor,
            'history': self.history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        # Save Keras model alongside
        model_path = path.with_suffix('.keras')
        self.model.save(model_path)
    
    @classmethod
    def load(cls, path: str) -> 'RiceClassifier':
        """
        Load classifier from disk.
        
        Args:
            path: File path to saved classifier
            
        Returns:
            Loaded classifier instance
        """
        path = Path(path)
        
        # Load state
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # Reconstruct classifier
        classifier = cls(
            features=state['features'],
            config=state['config']
        )
        classifier.preprocessor = state['preprocessor']
        classifier.history = state['history']
        
        # Load Keras model
        model_path = path.with_suffix('.keras')
        classifier.model = keras.models.load_model(model_path)
        
        return classifier
    
    def get_history(self) -> pd.DataFrame:
        """Get training history as DataFrame."""
        if self.history is None:
            raise ValueError("No training history. Call fit() first.")
        return pd.DataFrame(self.history)


# ============================================================================
# Experiment Utilities
# ============================================================================

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    random_state: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/validation/test sets.
    
    Args:
        df: Input dataframe
        train_ratio: Training fraction
        validation_ratio: Validation fraction
        random_state: Random seed
        
    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * validation_ratio)
    
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return (
        shuffled.iloc[:train_end],
        shuffled.iloc[train_end:val_end],
        shuffled.iloc[val_end:]
    )


def run_experiment(
    name: str,
    features: List[str],
    config: Optional[ClassifierConfig] = None,
    verbose: int = 1
) -> Tuple[RiceClassifier, Dict[str, float]]:
    """
    Run complete experiment: load data, split, train, evaluate.
    
    Args:
        name: Experiment name
        features: Features to use
        config: Hyperparameter config
        verbose: Training verbosity
        
    Returns:
        (classifier, results_dict)
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Features: {', '.join(features)}")
    print(f"{'='*60}")
    
    # Load and split data
    df = load_rice_data()
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=config.train_ratio if config else 0.8,
        validation_ratio=config.validation_ratio if config else 0.1,
        random_state=config.random_state if config else 100
    )
    
    # Train classifier
    classifier = RiceClassifier(features, config)
    classifier.fit(train_df, verbose=verbose)
    
    # Evaluate on all splits
    train_metrics = classifier.evaluate(train_df)
    val_metrics = classifier.evaluate(val_df)
    test_metrics = classifier.evaluate(test_df)
    
    # Print results
    print(f"\nRESULTS:")
    for split, metrics in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        print(f"  {split:5s}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
    
    # Aggregate results
    results = {
        f'train_{k}': v for k, v in train_metrics.items()
    }
    results.update({f'val_{k}': v for k, v in val_metrics.items()})
    results.update({f'test_{k}': v for k, v in test_metrics.items()})
    
    return classifier, results
