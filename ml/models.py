"""
The candidate model definitions.

Three candidates, as the assignment asks: a logistic-regression baseline, a
random forest, and gradient boosting. Each is wrapped in a scikit-learn
Pipeline together with its own preprocessing, so the artifact that gets saved
is the whole transform-and-predict chain - there is no separate preprocessing
step to forget at inference time.

XGBoost is used for the boosted candidate when it is installed and sklearn's
HistGradientBoostingClassifier otherwise, so the pipeline never fails just
because an optional dependency is missing.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import FEATURES

try:  # pragma: no cover - exercised by whichever branch the environment has
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

RANDOM_STATE = 42


def _numeric_preprocessor(scale: bool) -> ColumnTransformer:
    """Median-impute every feature; scale only where the learner needs it.

    avg_interpurchase_days is genuinely missing for single-order customers, so
    imputation is part of the model rather than something done to the data
    beforehand - that way the API applies exactly the same fill at inference.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("numeric", Pipeline(steps), FEATURES)])


def build_candidates(class_weight_ratio: float = 1.0) -> dict[str, Pipeline]:
    """The candidate grid, keyed by the name used in MLflow and the registry.

    ``class_weight_ratio`` is negatives/positives from the training block and
    is passed to XGBoost's scale_pos_weight; the other two learners take
    sklearn's ``class_weight="balanced"``, which computes the same correction
    internally.
    """
    candidates: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", _numeric_preprocessor(scale=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", _numeric_preprocessor(scale=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGBOOST:
        boosted = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=class_weight_ratio,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
        )
        candidates["xgboost"] = Pipeline(
            [("preprocessor", _numeric_preprocessor(scale=False)), ("classifier", boosted)]
        )
    else:  # pragma: no cover
        candidates["hist_gradient_boosting"] = Pipeline(
            [
                ("preprocessor", _numeric_preprocessor(scale=False)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_depth=5,
                        learning_rate=0.05,
                        max_iter=400,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    return candidates


def hyperparameters(pipeline: Pipeline) -> dict:
    """The classifier's own hyperparameters, for logging to MLflow."""
    classifier = pipeline.named_steps["classifier"]
    params = classifier.get_params()
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (int, float, str, bool, type(None)))
    }


def feature_importances(pipeline: Pipeline) -> dict[str, float] | None:
    """Per-feature importance or coefficient, where the learner exposes one."""
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
    else:  # pragma: no cover
        return None
    return {name: float(value) for name, value in zip(FEATURES, values)}
