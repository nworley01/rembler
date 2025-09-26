"""
Cross-validation utilities for pandas DataFrames.

This module provides functions for performing various types of cross-validation
on pandas DataFrames, including k-fold, stratified, and time series splits.
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional, Union, List, Dict, Any, Callable
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
    StratifiedGroupKFold
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def kfold_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Perform k-fold cross-validation split on a DataFrame.

    Args:
        df: Input DataFrame
        n_splits: Number of folds
        shuffle: Whether to shuffle data before splitting
        random_state: Random state for reproducibility

    Yields:
        Tuple of (train_df, test_df) for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        yield train_df, test_df


def stratified_kfold_split(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Perform stratified k-fold cross-validation split on a DataFrame.

    Args:
        df: Input DataFrame
        target_col: Name of the target column for stratification
        n_splits: Number of folds
        shuffle: Whether to shuffle data before splitting
        random_state: Random state for reproducibility

    Yields:
        Tuple of (train_df, test_df) for each fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_idx, test_idx in skf.split(df, df[target_col]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        yield train_df, test_df


def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    max_train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    gap: int = 0
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Perform time series cross-validation split on a DataFrame.

    Args:
        df: Input DataFrame (should be sorted by time)
        n_splits: Number of splits
        max_train_size: Maximum size for training set
        test_size: Size of test set
        gap: Number of samples to exclude between train and test

    Yields:
        Tuple of (train_df, test_df) for each fold
    """
    tss = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=max_train_size,
        test_size=test_size,
        gap=gap
    )

    for train_idx, test_idx in tss.split(df):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        yield train_df, test_df


def group_kfold_split(
    df: pd.DataFrame,
    group_col: str,
    n_splits: int = 5
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Perform group k-fold cross-validation split on a DataFrame.

    Args:
        df: Input DataFrame
        group_col: Name of the column containing group labels
        n_splits: Number of folds

    Yields:
        Tuple of (train_df, test_df) for each fold
    """
    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, test_idx in gkf.split(df, groups=df[group_col]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        yield train_df, test_df


def stratified_group_kfold_split(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Perform stratified group k-fold cross-validation split on a DataFrame.

    Args:
        df: Input DataFrame
        target_col: Name of the target column for stratification
        group_col: Name of the column containing group labels
        n_splits: Number of folds
        shuffle: Whether to shuffle groups before splitting
        random_state: Random state for reproducibility

    Yields:
        Tuple of (train_df, test_df) for each fold
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    for train_idx, test_idx in sgkf.split(df, df[target_col], df[group_col]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        yield train_df, test_df


def cross_validate_model(
    df: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    target_col: str,
    cv_method: str = 'kfold',
    n_splits: int = 5,
    scoring: Union[str, List[str], Dict[str, Callable]] = 'accuracy',
    **cv_kwargs
) -> Dict[str, List[float]]:
    """
    Perform cross-validation on a model using specified CV method.

    Args:
        df: Input DataFrame
        model: Model object with fit and predict methods
        feature_cols: List of feature column names
        target_col: Target column name
        cv_method: CV method ('kfold', 'stratified', 'timeseries', 'group', 'stratified_group')
        n_splits: Number of splits
        scoring: Scoring method(s) - can be string, list of strings, or dict of callables
        **cv_kwargs: Additional arguments for CV method

    Returns:
        Dictionary containing scores for each metric across folds
    """
    # Select CV method
    cv_methods = {
        'kfold': kfold_split,
        'stratified': stratified_kfold_split,
        'timeseries': time_series_split,
        'group': group_kfold_split,
        'stratified_group': stratified_group_kfold_split
    }

    if cv_method not in cv_methods:
        raise ValueError(f"cv_method must be one of {list(cv_methods.keys())}")

    cv_func = cv_methods[cv_method]

    # Prepare scoring functions
    if isinstance(scoring, str):
        scoring_funcs = {scoring: _get_scoring_func(scoring)}
    elif isinstance(scoring, list):
        scoring_funcs = {metric: _get_scoring_func(metric) for metric in scoring}
    elif isinstance(scoring, dict):
        scoring_funcs = scoring
    else:
        raise ValueError("scoring must be string, list of strings, or dict of callables")

    # Initialize results
    results = {metric: [] for metric in scoring_funcs.keys()}

    # Perform cross-validation
    cv_splits = cv_func(df, n_splits=n_splits, **cv_kwargs)

    for fold, (train_df, test_df) in enumerate(cv_splits):
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        # Fit model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate scores
        for metric_name, scoring_func in scoring_funcs.items():
            score = scoring_func(y_test, y_pred)
            results[metric_name].append(score)

    return results


def evaluate_cv_results(results: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Evaluate cross-validation results and return summary statistics.

    Args:
        results: Dictionary containing scores for each metric across folds

    Returns:
        DataFrame with mean, std, min, max for each metric
    """
    summary_data = {}

    for metric, scores in results.items():
        summary_data[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'scores': scores
        }

    summary_df = pd.DataFrame(summary_data).T
    return summary_df


def nested_cross_validation(
    df: pd.DataFrame,
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    feature_cols: List[str],
    target_col: str,
    outer_cv: str = 'kfold',
    inner_cv: str = 'kfold',
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = 'accuracy',
    **cv_kwargs
) -> Dict[str, Any]:
    """
    Perform nested cross-validation for model selection and evaluation.

    Args:
        df: Input DataFrame
        model_class: Model class to instantiate
        param_grid: Parameter grid for hyperparameter tuning
        feature_cols: List of feature column names
        target_col: Target column name
        outer_cv: Outer CV method
        inner_cv: Inner CV method
        outer_splits: Number of outer splits
        inner_splits: Number of inner splits
        scoring: Scoring metric
        **cv_kwargs: Additional CV arguments

    Returns:
        Dictionary with outer scores, best parameters for each fold, and summary
    """
    from sklearn.model_selection import GridSearchCV

    # Get CV methods
    cv_methods = {
        'kfold': lambda: KFold(n_splits=outer_splits, shuffle=True, random_state=42),
        'stratified': lambda: StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    }

    outer_cv_obj = cv_methods.get(outer_cv, cv_methods['kfold'])()
    inner_cv_obj = cv_methods.get(inner_cv, cv_methods['kfold'])()

    outer_scores = []
    best_params_per_fold = []

    X = df[feature_cols]
    y = df[target_col]

    for fold, (train_idx, test_idx) in enumerate(outer_cv_obj.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner CV for hyperparameter tuning
        grid_search = GridSearchCV(
            model_class(),
            param_grid,
            cv=inner_cv_obj,
            scoring=scoring,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        # Evaluate best model on outer test set
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)

        outer_scores.append(test_score)
        best_params_per_fold.append(grid_search.best_params_)

    return {
        'outer_scores': outer_scores,
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'best_params_per_fold': best_params_per_fold,
        'scoring_metric': scoring
    }


def _get_scoring_func(metric_name: str) -> Callable:
    """Get scoring function by name."""
    scoring_functions = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
    }

    if metric_name not in scoring_functions:
        raise ValueError(f"Unknown scoring metric: {metric_name}")

    return scoring_functions[metric_name]


def train_test_split_temporal(
    df: pd.DataFrame,
    time_col: str,
    test_size: float = 0.2,
    validation_size: Optional[float] = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Split DataFrame temporally based on time column.

    Args:
        df: Input DataFrame
        time_col: Column name containing time/date information
        test_size: Proportion of data for test set
        validation_size: Optional proportion for validation set

    Returns:
        Train/test split or train/validation/test split if validation_size provided
    """
    # Sort by time
    df_sorted = df.sort_values(time_col).copy()
    n_samples = len(df_sorted)

    if validation_size is not None:
        # Three-way split
        train_end = int(n_samples * (1 - test_size - validation_size))
        val_end = int(n_samples * (1 - test_size))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        return train_df, val_df, test_df
    else:
        # Two-way split
        split_idx = int(n_samples * (1 - test_size))

        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()

        return train_df, test_df


def rolling_window_validation(
    df: pd.DataFrame,
    window_size: int,
    step_size: int = 1,
    min_train_size: Optional[int] = None
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Perform rolling window validation on time series data.

    Args:
        df: Input DataFrame (should be sorted by time)
        window_size: Size of the test window
        step_size: Step size for rolling the window
        min_train_size: Minimum size for training set

    Yields:
        Tuple of (train_df, test_df) for each window
    """
    n_samples = len(df)

    for start in range(0, n_samples - window_size + 1, step_size):
        test_end = start + window_size

        if min_train_size and start < min_train_size:
            continue

        train_df = df.iloc[:start].copy()
        test_df = df.iloc[start:test_end].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            yield train_df, test_df