#!/usr/bin/env python3
"""
Train rice classifier and reproduce Colab experiments.

Usage:
    python train.py --experiment baseline    # 3 features, ~90% accuracy
    python train.py --experiment full        # 7 features, ~92% accuracy
    python train.py --experiment compare     # Compare both models
    python train.py --features Area Eccentricity --epochs 100  # Custom
"""

import argparse
import sys
from rice_classifier import (
    RiceClassifier, ClassifierConfig, ALL_FEATURES,
    load_rice_data, split_data, run_experiment
)


# Predefined experiments matching Colab
EXPERIMENTS = {
    'baseline': {
        'features': ['Eccentricity', 'Major_Axis_Length', 'Area'],
        'config': ClassifierConfig(threshold=0.35),
        'expected_acc': 0.90
    },
    'full': {
        'features': ALL_FEATURES,
        'config': ClassifierConfig(threshold=0.5),
        'expected_acc': 0.92
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train binary classifier for Turkish rice species'
    )
    
    parser.add_argument(
        '--experiment',
        choices=['baseline', 'full', 'compare'],
        help='Predefined experiment'
    )
    
    parser.add_argument(
        '--features',
        nargs='+',
        choices=ALL_FEATURES,
        help='Custom feature selection'
    )
    
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.35)
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--save', type=str, help='Path to save model')
    
    return parser.parse_args()


def run_comparison(verbose: int = 1):
    """Compare baseline and full models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    results = {}
    for name, exp in EXPERIMENTS.items():
        _, metrics = run_experiment(
            name.title(),
            exp['features'],
            exp['config'],
            verbose=verbose
        )
        results[name] = metrics
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Features':<5} {'Test Acc':<10} {'Test AUC':<10}")
    print("-" * 60)
    for name, metrics in results.items():
        n_feat = len(EXPERIMENTS[name]['features'])
        print(f"{name:<10} {n_feat:<5} {metrics['test_accuracy']:<10.4f} {metrics['test_auc']:<10.4f}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle comparison mode
    if args.experiment == 'compare':
        run_comparison(args.verbose)
        return 0
    
    # Determine configuration
    if args.experiment:
        exp = EXPERIMENTS[args.experiment]
        features = exp['features']
        config = exp['config']
        expected_acc = exp['expected_acc']
        name = args.experiment.title()
    elif args.features:
        features = args.features
        config = ClassifierConfig(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            threshold=args.threshold
        )
        expected_acc = None
        name = "Custom"
    else:
        print("Error: Must specify --experiment or --features")
        return 1
    
    # Run experiment
    classifier, results = run_experiment(name, features, config, args.verbose)
    
    # Validate against expected accuracy
    if expected_acc:
        test_acc = results['test_accuracy']
        diff = abs(test_acc - expected_acc)
        status = "✓ PASS" if diff < 0.05 else "⚠ WARN"
        print(f"\n{status}: Expected ~{expected_acc:.2f}, got {test_acc:.4f}")
    
    # Save model
    if args.save:
        classifier.save(args.save)
        print(f"\nModel saved to: {args.save}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
