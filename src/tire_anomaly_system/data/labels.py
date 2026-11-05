"""Fixed label order for existing four-state tire data and model checkpoints."""

ANOMALY_CLASS_NAMES = (
    "healthy",
    "tire_leak",
    "tire_overheat",
    "tire_imbalance",
)

ANOMALY_CLASS_MAP = {name: index for index, name in enumerate(ANOMALY_CLASS_NAMES)}


def validate_tire_label(label, anomaly_type):
    """Check recorded semantics without reinterpreting legacy class numbers."""
    if not anomaly_type:
        return
    expected = 0 if anomaly_type == "none" else ANOMALY_CLASS_MAP.get(anomaly_type)
    if expected is None:
        raise ValueError(
            f"Incompatible anomaly_type {anomaly_type!r}. Expected none/healthy, "
            "tire_leak, tire_overheat or tire_imbalance. Provide data labelled "
            "with the original project's current four-state schema; legacy "
            "pressure-relation and pressure-leak classes cannot be renamed to imbalance."
        )
    if label >= 0 and label != expected:
        raise ValueError(
            f"anomaly_class={label} conflicts with anomaly_type={anomaly_type!r}; "
            f"expected class {expected} ({ANOMALY_CLASS_NAMES[expected]})"
        )
