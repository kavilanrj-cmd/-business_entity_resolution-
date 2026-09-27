"""Central configuration for the entity-resolution pipeline.

Design goals
------------
* Every tunable lives here; no magic numbers scattered through the code base.
* No hard-coded dataset sizes -- all sizes are either limits derived from the
  data at runtime or explicit, overridable configuration values.
* No hard-coded country list -- countries are treated as an open-set string.
* The config can be round-tripped to/from YAML or JSON so the whole run is
  reproducible from a single file.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

#: Repository root = two levels above this file (``src/config.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Column names expected in the input TSV files.  The validator checks that the
#: provided data actually contains these before anything else runs.
ID_COLUMN = "entity_id"
NAME_COLUMN = "business_name"
ADDRESS_COLUMN = "business_address"
COUNTRY_COLUMN = "country"

SOURCE1 = "source1"
SOURCE2 = "source2"
SOURCE3 = "source3"

#: Repository root = two levels above this file (``src/config.py``).
#:
#: The real Amazon ML Challenge data lives under
#: ``dataset/student_resource/dataset``; ``dataset/`` itself only holds the
#: synthetic fixture from ``tests/make_fixture.py``.  The loader auto-detects
#: the real root -- see :func:`src.data.loader.resolve_dataset_root` -- and these
#: defaults follow it so the CLI entry points cannot silently use the fixture.
DEFAULT_DATASET_ROOT: Path = PROJECT_ROOT / "dataset" / "student_resource" / "dataset"


@dataclass
class PathConfig:
    """Filesystem locations.  All paths are relative to ``base_dir``."""

    base_dir: str = "."
    train_dir: str = str(DEFAULT_DATASET_ROOT / "train")
    test_dir: str = str(DEFAULT_DATASET_ROOT / "test")
    models_dir: str = "models"
    output_dir: str = "output"
    reports_dir: str = "reports"

    def _resolve(self, base: Path) -> "PathConfig":
        return PathConfig(
            base_dir=str(base),
            train_dir=str(self._abs(self.train_dir, base)),
            test_dir=str(self._abs(self.test_dir, base)),
            models_dir=str(self._abs(self.models_dir, base)),
            output_dir=str(self._abs(self.output_dir, base)),
            reports_dir=str(self._abs(self.reports_dir, base)),
        )

    @staticmethod
    def _abs(value: str, base: Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p)

    def ensure_dirs(self) -> None:
        for attr in ("models_dir", "output_dir", "reports_dir"):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessConfig:
    """Switches for text normalization."""

    #: Strip legal-form suffixes / abbreviations from names.
    normalize_legal_forms: bool = True
    #: Collapse ``&`` into ``and`` (or vice versa) instead of deleting it.
    ampersand_to_and: bool = True
    #: Drop punctuation entirely rather than translating it to spaces.
    drop_punctuation: bool = True
    #: Remove single-character filler tokens (e.g. ``&`` leftovers, "a").
    drop_single_char_tokens: bool = False
    #: Unicode transliteration table applied before lower-casing.
    use_unicode_translit: bool = True
    #: Extract address components (pincode / house number / city / state).
    extract_address_components: bool = True


@dataclass
class BlockingConfig:
    """Multi-strategy candidate generation.

    All caps are *upper bounds per Source 1 entity*; they bound runtime and
    memory.  Their effect on candidate recall is measured and reported by
    :mod:`src.evaluation.validation`, never assumed.
    """

    enabled: tuple[str, ...] = (
        "exact_name",
        "token_name",
        "tfidf_name",
        "address_token",
        "country_aware",
    )

    # --- Strategy A: exact normalized name -------------------------------
    exact_name_max_per_s1: int = 200

    # --- Strategy B: IDF-weighted name token overlap ---------------------
    token_name_min_score: float = 0.30
    token_name_max_per_s1: int = 150

    # --- Strategy C: character n-gram TF-IDF -----------------------------
    tfidf_name_analyzer: str = "char_wb"
    tfidf_name_ngram_min: int = 2
    tfidf_name_ngram_max: int = 5
    tfidf_name_top_k: int = 40
    tfidf_name_min_score: float = 0.20

    # --- Strategy D: address token blocking ------------------------------
    address_token_min_score: float = 0.35
    address_token_max_per_s1: int = 100

    # --- Strategy E: country-aware retrieval -----------------------------
    #: Also retrieve within the country-restricted pool (open-set, derived
    #: from the data -- never from a hard-coded country list).
    country_aware_top_k: int = 15
    country_aware_min_score: float = 0.25
    country_aware_min_pool_size: int = 5
    #: If a Source 1 record has a missing country, fall back to a global
    #: retrieval so that open-set / missing countries are still covered.
    country_aware_fallback_top_k: int = 20

    # --- Union / post-processing -----------------------------------------
    #: Drop this many fraction of the weakest candidates per Source 1 entity
    #: (0 disables the safety-net pruning).
    relative_prune_ratio: float = 0.0
    #: Absolute per-entity cap applied to the *union* of all strategies.
    #: ``None`` means "keep everything the strategies produced".
    max_candidates_per_s1: int | None = 400
    #: Minimum number of candidates kept per Source 1 entity regardless of
    #: score, so that hard cases are not left with zero options.
    min_candidates_per_s1: int = 5


@dataclass
class FeatureConfig:
    """Feature-engineering switches."""

    #: Character n-gram range for the *name* TF-IDF used as a feature.
    name_tfidf_ngram_min: int = 2
    name_tfidf_ngram_max: int = 5
    address_tfidf_ngram_min: int = 2
    address_tfidf_ngram_max: int = 4
    #: Order of the Jaro-Winkler prefix bonus (4 is the standard).
    jaro_winkler_prefix_weight: float = 0.1
    #: Length of the raw-character prefix used for the "prefix match" feature.
    prefix_chars: int = 4
    #: Include heuristic city/state components (gated on *both* sides being
    #: present, so weak extraction cannot manufacture false evidence).
    use_address_components: bool = True
    #: Include engineered ratio / combined features.
    use_combined_features: bool = True


@dataclass
class ModelConfig:
    """Which candidate models to train, and how."""

    candidates: tuple[str, ...] = ("logistic_regression", "random_forest", "hist_gradient_boosting")
    random_state: int = 42
    #: Number of Source 1 entities held out for validation (fraction when
    #: ``< 1``, absolute count otherwise).
    validation_size: float = 0.25
    #: Optional second held-out fold used for model selection reporting.
    n_splits: int = 4
    max_iter: int = 2000
    rf_n_estimators: int = 400
    rf_min_samples_leaf: int = 2
    rf_max_features: float = 0.5
    hgb_max_iter: int = 400
    hgb_learning_rate: float = 0.06
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
    hgb_l2_regularization: float = 1.0
    n_jobs: int = -1
    #: Refit the winning model on 100% of the training Source 1 entities.
    refit_on_full_train: bool = True


@dataclass
class ThresholdConfig:
    """Score-threshold selection for the F0.5-optimizing decision rule."""

    #: The coarse grid spans nearly the whole probability range.  A narrow
    #: high-end grid silently hides the optimum whenever the model is
    #: well-calibrated-but-conservative and the best cut sits far below 0.3.
    grid_start: float = 0.01
    grid_stop: float = 0.99
    grid_step: float = 0.01
    #: Extra candidate thresholds derived from the observed validation score
    #: distribution (exact breakpoints) -- these often beat the coarse grid.
    use_score_breakpoints: bool = True
    max_breakpoints: int = 400
    #: Guard rails: if a candidate threshold yields macro precision below
    #: this, it is rejected in favour of a more conservative one.
    min_macro_precision: float = 0.0
    #: How far above the best threshold we are willing to move when
    #: precision-first tie-breaking applies (F0.5 is precision weighted).
    tie_break_eps: float = 1e-9


@dataclass
class OutputConfig:
    matching_results_file: str = "matching_results.tsv"
    candidate_pairs_file: str = "candidate_pairs.tsv"
    #: Emit an empty ``matched_entity_ids`` cell (not the literal "[]").
    empty_placeholder: str = ""
    candidate_score_file: str = "candidate_scores.tsv"


@dataclass
class Config:
    """Root configuration object."""

    paths: PathConfig = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    random_state: int = 42
    log_level: str = "INFO"
    #: Path of the official ``validate_submission.py`` if the organizer
    #: shipped one.  Auto-detected under ``utils/`` when it exists.
    official_validator: str | None = None

    # -- serialisation ----------------------------------------------------
    def ensure_dirs(self) -> None:
        """Create the models / output / reports directories."""
        self.paths.ensure_dirs()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore

                path.write_text(
                    yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
                    encoding="utf-8",
                )
                return
            except ImportError:  # pragma: no cover - optional dependency
                LOGGER.warning("PyYAML unavailable; writing JSON syntax into %s", path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> "Config":
        """Build a config from an optional YAML/JSON file plus overrides.

        Overrides use dotted paths, e.g. ``{"blocking.tfidf_name_top_k": 60}``.
        """
        data: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {p}")
            text = p.read_text(encoding="utf-8")
            if p.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml  # type: ignore

                    data = yaml.safe_load(text) or {}
                except ImportError:  # pragma: no cover - optional dependency
                    data = json.loads(text)
            else:
                data = json.loads(text)
        if overrides:
            data = _deep_update(data, dict(overrides))

        cfg = _from_mapping(cls, data)
        base = Path(data.get("paths", {}).get("base_dir", "."))
        if not base.is_absolute():
            base = (Path(path).resolve().parent / base) if path is not None else (PROJECT_ROOT / base)
        cfg.paths = cfg.paths._resolve(base)
        return cfg


def _deep_update(base: dict[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in other.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion of a config value to the annotated type."""
    origin = getattr(target_type, "__origin__", None)
    if origin is not None:
        args = [a for a in getattr(target_type, "__args__", ()) if a is not type(None)]  # noqa: E721
        if value is None:
            return None
        if origin in (tuple, Sequence):
            if isinstance(value, str):
                value = [value]
            return tuple(value)
        if origin is str or target_type is str:
            return str(value)
        if origin is bool:
            return bool(value)
        if origin in (int, float):
            return origin(value)  # type: ignore[operator]
        if origin is tuple and args:
            return _coerce(value, args[0])
        return value
    if target_type is bool and not isinstance(value, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if target_type is int and not isinstance(value, bool):
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str and value is not None:
        return str(value)
    return value


def _from_mapping(cls: Any, data: Mapping[str, Any]) -> Any:
    """Recursively instantiate nested dataclasses from a mapping."""
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) or (isinstance(f.type, type) and is_dataclass(f.type)):
            kwargs[f.name] = _from_mapping(f.type, value)  # type: ignore[arg-type]
        elif isinstance(f.type, str) and f.type in _KNOWN_SECTIONS:
            kwargs[f.name] = _from_mapping(_KNOWN_SECTIONS[f.type], value)
        else:
            kwargs[f.name] = _coerce(value, f.type)
    return cls(**kwargs)


_KNOWN_SECTIONS: dict[str, Any] = {
    "PathConfig": PathConfig,
    "PreprocessConfig": PreprocessConfig,
    "BlockingConfig": BlockingConfig,
    "FeatureConfig": FeatureConfig,
    "ModelConfig": ModelConfig,
    "ThresholdConfig": ThresholdConfig,
    "OutputConfig": OutputConfig,
}


def setup_logging(level: str = "INFO") -> None:
    """Configure a single, non-duplicating logging handler."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-38s | %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def ensure_required(package: str, extra: str) -> None:
    """Fail loudly with an actionable message for a missing dependency."""
    try:
        __import__(package)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"Missing optional dependency '{package}'. Install with: pip install {extra}"
        ) from exc
