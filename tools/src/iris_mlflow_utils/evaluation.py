"""Model evaluation helpers shared by the training notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize  # type: ignore[import-untyped]


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and diagnostics for one model/data partition."""

    metrics: dict[str, float]
    report: dict[str, Any]
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None = None


def evaluate_model(
    model: Any, features: Any, target: np.ndarray, labels: list[int]
) -> EvaluationResult:
    """Evaluate a fitted classifier and return metrics plus diagnostics."""

    predictions = np.asarray(model.predict(features))
    probabilities = None
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = np.asarray(predict_proba(features))
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        target, predictions, average="weighted", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        target, predictions, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_macro": float(precision_macro),
        "precision_weighted": float(precision_weighted),
        "recall_macro": float(recall_macro),
        "recall_weighted": float(recall_weighted),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    return EvaluationResult(
        metrics=metrics,
        report=classification_report(
            target, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        confusion_matrix=confusion_matrix(target, predictions, labels=labels),
        predictions=predictions,
        probabilities=probabilities,
    )


def evaluate_train_test(
    model: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_count: int,
) -> dict[str, EvaluationResult]:
    """Evaluate a fitted classifier consistently on train and test partitions."""

    labels = list(range(class_count))
    return {
        "train": evaluate_model(model, x_train, y_train, labels),
        "test": evaluate_model(model, x_test, y_test, labels),
    }


def flatten_metrics(results: dict[str, EvaluationResult]) -> dict[str, float]:
    """Prefix partition names so metrics are unambiguous in MLflow."""

    return {
        f"{partition}_{name}": value
        for partition, result in results.items()
        for name, value in result.metrics.items()
    }


def build_metrics_summary_table(
    results: dict[str, EvaluationResult],
    *,
    model_type: str,
    dataset_version: str,
    project_version: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Build an MLflow table with one row per partition and metric."""

    rows = [
        {
            "model_type": model_type,
            "partition": partition,
            "metric": metric,
            "value": value,
            "dataset_version": dataset_version,
            "project_version": project_version,
            "run_id": run_id,
        }
        for partition, result in results.items()
        for metric, value in result.metrics.items()
    ]
    return pd.DataFrame(rows)


def build_classification_table(
    results: dict[str, EvaluationResult],
    classes: tuple[str, ...],
    *,
    model_type: str,
    dataset_version: str,
    project_version: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Normalize per-class reports into a table suitable for MLflow."""

    rows: list[dict[str, Any]] = []
    for partition, result in results.items():
        for class_id, class_name in enumerate(classes):
            values = result.report.get(str(class_id), {})
            rows.append(
                {
                    "model_type": model_type,
                    "partition": partition,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": float(values.get("precision", 0.0)),
                    "recall": float(values.get("recall", 0.0)),
                    "f1_score": float(values.get("f1-score", 0.0)),
                    "support": int(values.get("support", 0)),
                    "dataset_version": dataset_version,
                    "project_version": project_version,
                    "run_id": run_id,
                }
            )
    return pd.DataFrame(rows)


def _lift_frame(y_true: np.ndarray, scores: np.ndarray, class_id: int) -> pd.DataFrame:
    """Build decile lift and cumulative gain data for one-vs-rest scores."""

    frame = pd.DataFrame({"actual": (y_true == class_id).astype(int), "score": scores})
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["population_fraction"] = frame["rank"] / len(frame)
    frame["cumulative_positive_rate"] = frame["actual"].cumsum() / frame["rank"]
    baseline = frame["actual"].mean()
    frame["lift"] = frame["cumulative_positive_rate"] / baseline if baseline else 0.0
    frame["class_id"] = class_id
    return frame


def build_evaluation_artifacts(
    model: Any,
    result: EvaluationResult,
    *,
    labels: list[int],
    class_names: tuple[str, ...],
    features: pd.DataFrame,
    target: np.ndarray,
    output_dir: Path,
) -> dict[str, Path]:
    """Create standard classifier diagnostics for MLflow logging.

    ROC, precision-recall, lift and gain are generated one-vs-rest for each
    class and use macro/micro aggregates where probabilities are available.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    confusion = result.confusion_matrix
    for normalized, filename, title, values in (
        (False, "confusion_matrix.png", "Matriz de confusión", confusion),
        (
            True,
            "confusion_matrix_normalized.png",
            "Matriz de confusión normalizada",
            confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1),
        ),
    ):
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        figure.colorbar(image, ax=axis)
        axis.set(
            title=title,
            xlabel="Predicción",
            ylabel="Real",
            xticks=labels,
            yticks=labels,
            xticklabels=class_names,
            yticklabels=class_names,
        )
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                text = f"{values[row, column]:.2f}" if normalized else str(values[row, column])
                axis.text(column, row, text, ha="center", va="center")
        path = output_dir / filename
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths[filename] = path

    probabilities = result.probabilities
    if probabilities is None or probabilities.shape[1] != len(labels):
        return paths
    binary_target = label_binarize(target, classes=labels)
    roc_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    figure, axis = plt.subplots(figsize=(7, 6))
    for class_id, class_name in zip(labels, class_names, strict=True):
        false_positive, true_positive, _ = roc_curve(
            binary_target[:, class_id], probabilities[:, class_id]
        )
        roc_curves[class_id] = (false_positive, true_positive)
        axis.plot(false_positive, true_positive, label=class_name)
    micro_false_positive, micro_true_positive, _ = roc_curve(
        binary_target.ravel(), probabilities.ravel()
    )
    axis.plot(micro_false_positive, micro_true_positive, linestyle="--", label="micro-average")
    axis.plot([0, 1], [0, 1], "k:")
    axis.set(
        title="Curva ROC one-vs-rest", xlabel="False Positive Rate", ylabel="True Positive Rate"
    )
    axis.legend()
    path = output_dir / "roc_curve_by_class.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths[path.name] = path

    all_false_positive = np.unique(np.concatenate([curve[0] for curve in roc_curves.values()]))
    mean_true_positive = np.zeros_like(all_false_positive)
    for false_positive, true_positive in roc_curves.values():
        mean_true_positive += np.interp(all_false_positive, false_positive, true_positive)
    mean_true_positive /= len(roc_curves)

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        all_false_positive,
        mean_true_positive,
        label="macro-average",
    )
    axis.plot(micro_false_positive, micro_true_positive, linestyle="--", label="micro-average")
    axis.plot([0, 1], [0, 1], "k:")
    axis.set(title="ROC macro/micro", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend()
    path = output_dir / "roc_curve_macro_micro.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths[path.name] = path

    figure, axis = plt.subplots(figsize=(7, 6))
    for class_id, class_name in zip(labels, class_names, strict=True):
        precision, recall, _ = precision_recall_curve(
            binary_target[:, class_id], probabilities[:, class_id]
        )
        axis.plot(recall, precision, label=class_name)
    axis.set(title="Curva Precision-Recall", xlabel="Recall", ylabel="Precision")
    axis.legend()
    path = output_dir / "precision_recall_curve.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths[path.name] = path

    lift_frames = [_lift_frame(target, probabilities[:, class_id], class_id) for class_id in labels]
    lift = pd.concat(lift_frames, ignore_index=True)
    lift_path = output_dir / "lift_data.json"
    lift.to_json(lift_path, orient="records")
    paths[lift_path.name] = lift_path
    gain = lift.copy()
    positives_by_class = gain.groupby("class_id")["actual"].transform("sum").clip(lower=1)
    gain["cumulative_gain"] = gain.groupby("class_id")["actual"].cumsum() / positives_by_class
    gain_path = output_dir / "cumulative_gain_data.json"
    gain.to_json(gain_path, orient="records")
    paths[gain_path.name] = gain_path
    for filename, title, y_column in (
        ("lift_curve_by_class.png", "Lift por clase", "lift"),
        ("cumulative_gain_curve.png", "Cumulative gain", "cumulative_gain"),
    ):
        figure, axis = plt.subplots(figsize=(7, 6))
        for class_id, class_name in zip(labels, class_names, strict=True):
            subset = gain[gain["class_id"] == class_id]
            axis.plot(subset["population_fraction"], subset[y_column], label=class_name)
        axis.set(title=title, xlabel="Fracción de población", ylabel=y_column)
        axis.legend()
        path = output_dir / filename
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths[filename] = path
    macro_path = output_dir / "lift_curve_macro.json"
    macro_lift = lift.groupby("population_fraction", as_index=False)["lift"].mean()
    macro_lift.to_json(macro_path, orient="records")
    paths[macro_path.name] = macro_path
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot(macro_lift["population_fraction"], macro_lift["lift"], label="macro-average")
    axis.axhline(1.0, color="black", linestyle=":", label="baseline")
    axis.set(title="Lift macro", xlabel="Fracción de población", ylabel="lift")
    axis.legend()
    path = output_dir / "lift_curve_macro.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths[path.name] = path

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(np.max(probabilities, axis=1), bins=10, alpha=0.8)
    axis.set(
        title="Distribución de probabilidades", xlabel="Probabilidad máxima", ylabel="Frecuencia"
    )
    path = output_dir / "probability_distribution.png"
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    paths[path.name] = path

    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        importance = getattr(model, "feature_importances", None)
    if importance is not None:
        importance_frame = pd.DataFrame(
            {"feature": list(features.columns), "importance": np.asarray(importance)}
        ).sort_values("importance", ascending=False)
        importance_path = output_dir / "feature_importance.json"
        importance_frame.to_json(importance_path, orient="records")
        paths[importance_path.name] = importance_path
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.barh(importance_frame["feature"], importance_frame["importance"])
        axis.invert_yaxis()
        axis.set(title="Feature importance", xlabel="Importance")
        path = output_dir / "feature_importance.png"
        figure.tight_layout()
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths[path.name] = path
    return paths


def build_probability_metrics(
    result: EvaluationResult, target: np.ndarray, labels: list[int]
) -> dict[str, float]:
    """Return multiclass ROC-AUC and average precision metrics."""

    if result.probabilities is None:
        return {}
    binary_target = label_binarize(target, classes=labels)
    return {
        "log_loss": float(log_loss(target, result.probabilities, labels=labels)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(binary_target, result.probabilities, multi_class="ovr", average="macro")
        ),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(
                binary_target, result.probabilities, multi_class="ovr", average="weighted"
            )
        ),
        "average_precision_macro": float(
            average_precision_score(binary_target, result.probabilities, average="macro")
        ),
        "average_precision_weighted": float(
            average_precision_score(binary_target, result.probabilities, average="weighted")
        ),
    }
