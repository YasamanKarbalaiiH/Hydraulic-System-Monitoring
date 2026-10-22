from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "outputs"

SENSOR_NAMES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4",
    "VS1", "CE", "CP", "SE"
]
FEATURE_TYPES = ["mean", "min", "max", "std", "rms"]
PROFILE_NAMES = [
    "Cooler", "Valve", "Pump leakage", "Accumulator pressure", "Stability"
]
IDEAL_PROFILE = np.array([100.0, 100.0, 0.0, 130.0, 0.0])


def extract_features(sensor_data):
    """Extract features from raw data."""
    sensor_data = np.asarray(sensor_data, dtype=float)
    return np.column_stack((
        np.mean(sensor_data, axis=1),
        np.min(sensor_data, axis=1),
        np.max(sensor_data, axis=1),
        np.std(sensor_data, axis=1),
        np.sqrt(np.mean(np.square(sensor_data), axis=1))
    ))


def load_dataset(dataset_dir=DATASET_DIR):
    """Load and validate all sensor files and the equipment profile."""
    sensor_data = {}
    feature_blocks = []
    expected_cycles = 2205

    for sensor_name in SENSOR_NAMES:
        path = Path(dataset_dir) / f"{sensor_name}.txt"

        if not path.exists():
            raise FileNotFoundError(
                f"فایل سنسور پیدا نشد: {path}"
            )

        values = np.loadtxt(path)

        if values.ndim != 2:
            raise ValueError(
                f"فایل {sensor_name} باید آرایه دوبعدی داشته باشد."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"فایل {sensor_name} شامل NaN یا Inf است."
            )

        if values.shape[0] != expected_cycles:
            raise ValueError(
                f"فایل {sensor_name} باید {expected_cycles} چرخه داشته باشد؛ "
                f"تعداد فعلی: {values.shape[0]}."
            )

        if values.shape[1] == 0:
            raise ValueError(
                f"فایل {sensor_name} هیچ نمونه‌ای در چرخه‌ها ندارد."
            )

        sensor_data[sensor_name] = values
        feature_blocks.append(extract_features(values))

    features = np.concatenate(feature_blocks, axis=1)

    feature_names = [
        f"{sensor}_{feature}"
        for sensor in SENSOR_NAMES
        for feature in FEATURE_TYPES
    ]

    profile_path = Path(dataset_dir) / "profile.txt"

    if not profile_path.exists():
        raise FileNotFoundError(
            f"فایل وضعیت تجهیزات پیدا نشد: {profile_path}"
        )

    profile = np.loadtxt(profile_path)
    

    if profile.shape != (expected_cycles, len(PROFILE_NAMES)):
        raise ValueError(
            f"ابعاد profile باید "
            f"({expected_cycles}, {len(PROFILE_NAMES)}) باشد؛ "
            f"ابعاد فعلی: {profile.shape}."
        )

    if not np.isfinite(profile).all():
        raise ValueError(
            "فایل profile شامل NaN یا Inf است."
        )

    return {
        "sensor_data": sensor_data,
        "features": features,
        "feature_names": feature_names,
        "profile": profile,
        "samples_per_cycle": {
            name: values.shape[1]
            for name, values in sensor_data.items()
        },
    }
def prepare_comparison(features, profile=None, random_seed=42, train_fraction=0.8):
    """Split rows reproducibly, reserving healthy cycles for both train and test."""
    features = np.asarray(features, dtype=float)
    rng = np.random.default_rng(random_seed)
    split_point = int(len(features) * train_fraction)
    test_count = len(features) - split_point

    if profile is not None:
        profile = np.asarray(profile, dtype=float)
        if profile.shape != (len(features), len(PROFILE_NAMES)):
            raise ValueError("ابعاد profile با تعداد چرخه‌های ویژگی‌ها سازگار نیست.")
        fully_healthy = np.all(np.isclose(profile, IDEAL_PROFILE), axis=1)
        healthy_indices = np.flatnonzero(fully_healthy)
        other_indices = np.flatnonzero(~fully_healthy)

        # Keep at least one fully healthy cycle in each split when possible,
        # while leaving at least two healthy training cycles for calibration.
        if len(healthy_indices) >= 3 and test_count > 0:
            healthy_indices = rng.permutation(healthy_indices)
            other_indices = rng.permutation(other_indices)
            healthy_test_count = min(
                len(healthy_indices) - 2,
                max(1, int(round(len(healthy_indices) * (1 - train_fraction))))
            )
            healthy_test = healthy_indices[:healthy_test_count]
            healthy_train = healthy_indices[healthy_test_count:]
            other_test_count = test_count - len(healthy_test)
            if other_test_count < 0 or other_test_count > len(other_indices):
                raise ValueError("تقسیم طبقه‌بندی‌شده داده‌ها ممکن نیست.")
            test_other = other_indices[:other_test_count]
            train_other = other_indices[other_test_count:]
            train_indices = rng.permutation(np.concatenate((healthy_train, train_other)))
            test_indices = rng.permutation(np.concatenate((healthy_test, test_other)))
        else:
            indices = rng.permutation(len(features))
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
    else:
        indices = rng.permutation(len(features))
        train_indices = indices[:split_point]
        test_indices = indices[split_point:]

    X_train = features[train_indices]
    X_test = features[test_indices]
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    safe_std = np.where(train_std == 0, 1.0, train_std)

    X_train_scaled = (X_train - train_mean) / safe_std
    X_test_scaled = (X_test - train_mean) / safe_std

    params = pd.DataFrame({
        "feature": np.arange(features.shape[1]),
        "training_mean": train_mean,
        "training_std": train_std,
        "std_used": safe_std,
        "constant_in_training": train_std == 0,
    })

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "train_mean": train_mean,
        "train_std": train_std,
        "safe_std": safe_std,
        "standardization_params": params,
    }


def calculate_correlations(X_train, feature_names, threshold=0.90):
    """Calculate correlations safely, excluding constant columns from pair ranking."""
    X_train = np.asarray(X_train, dtype=float)
    stds = np.std(X_train, axis=0)
    variable_mask = stds > 0
    variable_indices = np.flatnonzero(variable_mask)
    pairs = []

    if len(variable_indices) >= 2:
        corr = np.corrcoef(X_train[:, variable_indices], rowvar=False)
        corr = np.atleast_2d(corr)
        for i in range(len(variable_indices)):
            for j in range(i + 1, len(variable_indices)):
                value = corr[i, j]
                if np.isfinite(value) and abs(value) >= threshold:
                    left = int(variable_indices[i])
                    right = int(variable_indices[j])
                    pairs.append({
                        "feature_1": feature_names[left],
                        "feature_2": feature_names[right],
                        "correlation": float(value),
                        "absolute_correlation": float(abs(value)),
                    })

    pairs_df = pd.DataFrame(
        pairs,
        columns=["feature_1", "feature_2", "correlation", "absolute_correlation"]
    )
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            "absolute_correlation", ascending=False
        ).reset_index(drop=True)

    return pairs_df, variable_mask


def select_independent_features(X_train, feature_names):
    """Greedily keep columns that increase numerical matrix rank."""
    X_train = np.asarray(X_train, dtype=float)
    kept = []
    rank = 0
    for column_index in range(X_train.shape[1]):
        candidate = kept + [column_index]
        candidate_rank = np.linalg.matrix_rank(X_train[:, candidate])
        if candidate_rank > rank:
            kept.append(column_index)
            rank = candidate_rank

    constant_columns = [
        feature_names[i] for i in range(X_train.shape[1])
        if np.std(X_train[:, i]) == 0
    ]
    return {
        "independent_indices": np.array(kept, dtype=int),
        "independent_names": [feature_names[i] for i in kept],
        "matrix_rank": int(np.linalg.matrix_rank(X_train)),
        "constant_columns": constant_columns,
    }


def find_similar_cycles(query_cycle_id, train_indices, test_indices,
                        X_train_scaled, X_test_scaled, top_n=5):
    """Compare a test cycle with all training cycles using distance and cosine similarity."""
    cycle_zero_index = int(query_cycle_id) - 1
    match_positions = np.flatnonzero(test_indices == cycle_zero_index)
    if len(match_positions) == 0:
        raise ValueError("چرخه انتخاب‌شده در مجموعه آزمون نیست.")
    query_position = int(match_positions[0])
    query = X_test_scaled[query_position]

    distances = np.linalg.norm(X_train_scaled - query, axis=1)
    train_norms = np.linalg.norm(X_train_scaled, axis=1)
    query_norm = np.linalg.norm(query)
    denominator = train_norms * query_norm
    dot_products = X_train_scaled @ query
    cosine = np.divide(
        dot_products, denominator,
        out=np.zeros_like(dot_products, dtype=float),
        where=denominator != 0
    )

    nearest = np.argsort(distances)[:top_n]
    most_cosine = np.argsort(cosine)[::-1][:top_n]

    distance_df = pd.DataFrame({
        "cycle_id": train_indices[nearest] + 1,
        "euclidean_distance": distances[nearest],
        "cosine_similarity": cosine[nearest],
    })
    cosine_df = pd.DataFrame({
        "cycle_id": train_indices[most_cosine] + 1,
        "cosine_similarity": cosine[most_cosine],
        "euclidean_distance": distances[most_cosine],
    })
    return {
        "query_cycle_id": int(query_cycle_id),
        "query_scaled": query,
        "distances": distances,
        "cosine_similarities": cosine,
        "nearest_positions": nearest,
        "cosine_positions": most_cosine,
        "distance_df": distance_df,
        "cosine_df": cosine_df,
    }


def calculate_cycle_changes(X_scaled, feature_names, top_n=10):
    """Rank adjacent cycle pairs by Euclidean distance in standardized feature space."""
    if len(X_scaled) < 2:
        return pd.DataFrame(columns=[
            "cycle_from", "cycle_to", "change_distance", "largest_change_feature"
        ])
    differences = np.diff(X_scaled, axis=0)
    distances = np.linalg.norm(differences, axis=1)
    largest_feature_indices = np.argmax(np.abs(differences), axis=1)
    order = np.argsort(distances)[::-1][:top_n]
    return pd.DataFrame({
        "cycle_from": order + 1,
        "cycle_to": order + 2,
        "change_distance": distances[order],
        "largest_change_feature": [
            feature_names[largest_feature_indices[i]] for i in order
        ],
    })


def calculate_healthy_distances(features, profile, train_indices, test_indices,
                                X_train_scaled, X_test_scaled):
    """Build an ideal-profile reference from healthy training cycles only."""
    fully_healthy = np.all(np.isclose(profile, IDEAL_PROFILE), axis=1)
    train_healthy_positions = np.flatnonzero(fully_healthy[train_indices])

    if len(train_healthy_positions) == 0:
        raise ValueError(
            "در مجموعه آموزش چرخه کاملاً سالمی با پروفایل ایده‌آل پیدا نشد؛ "
            "ساخت مرجع سالم ممکن نیست."
        )

    healthy_reference = np.mean(
        X_train_scaled[train_healthy_positions], axis=0
    )
    all_scaled = np.empty_like(features, dtype=float)
    all_scaled[train_indices] = X_train_scaled
    all_scaled[test_indices] = X_test_scaled
    distances = np.linalg.norm(all_scaled - healthy_reference, axis=1)

    return {
        "fully_healthy_mask": fully_healthy,
        "train_healthy_positions": train_healthy_positions,
        "healthy_train_cycle_ids": train_indices[train_healthy_positions] + 1,
        "healthy_reference": healthy_reference,
        "all_scaled": all_scaled,
        "healthy_distances": distances,
        "healthy_cycle_count": len(train_healthy_positions),
    }


def determine_thresholds(healthy_info, train_indices, percentile_95=95, percentile_99=99):
    """Estimate thresholds from leave-one-out distances for healthy training rows."""
    healthy_ids = healthy_info["healthy_train_cycle_ids"]
    healthy_reference = healthy_info["healthy_reference"]
    all_scaled = healthy_info["all_scaled"]
    train_healthy_positions = healthy_info["train_healthy_positions"]
    # The reference is recalculated without each healthy row to avoid zero-biased
    # self-distance. With very few healthy examples, these estimates are uncertain.
    leave_one_out = []
    for position in train_healthy_positions:
        healthy_rows = np.delete(train_healthy_positions, np.where(train_healthy_positions == position)[0][0])
        if len(healthy_rows) == 0:
            continue
        # healthy rows here are positions within X_train_scaled, while the reference
        # vector is in the same scaled feature space.
        healthy_train_scaled = all_scaled[train_indices]
        reference_without_one = np.mean(healthy_train_scaled[healthy_rows], axis=0)
        leave_one_out.append(np.linalg.norm(healthy_train_scaled[position] - reference_without_one))

    if len(leave_one_out) < 2:
        # Fallback is explicit and flagged: there is not enough data to estimate a
        # stable healthy distribution.
        healthy_distances = healthy_info["healthy_distances"][healthy_info["fully_healthy_mask"]]
        if len(healthy_distances) == 0:
            raise ValueError("برای محاسبه آستانه، داده سالم کافی وجود ندارد.")
        calibration = healthy_distances
        calibration_method = "فاصله چرخه‌های سالم تا مرجع؛ تعداد نمونه کم"
    else:
        calibration = np.asarray(leave_one_out, dtype=float)
        calibration_method = "فاصله leave-one-out چرخه‌های سالم آموزش"

    threshold_95 = float(np.percentile(calibration, percentile_95))
    threshold_99 = float(np.percentile(calibration, percentile_99))
    return {
        "threshold_95": threshold_95,
        "threshold_99": threshold_99,
        "calibration_sample_count": int(len(calibration)),
        "calibration_method": calibration_method,
        "calibration_distances": np.asarray(calibration),
        "healthy_cycle_count": int(len(healthy_ids)),
    }


def classify_alerts(distances, threshold_95, threshold_99):
    """Assign transparent distance-based alert levels."""
    if threshold_99 < threshold_95:
        threshold_95, threshold_99 = threshold_99, threshold_95
    labels = np.where(
        distances <= threshold_95, "Normal",
        np.where(distances <= threshold_99, "Suspicious", "Needs inspection")
    )
    return labels


def evaluate_alerts(profile, test_indices, healthy_distances, threshold_95, threshold_99):
    """Evaluate equipment-fault alerts on stable test cycles; report stability separately."""
    alerts = classify_alerts(
        healthy_distances[test_indices], threshold_95, threshold_99
    )
    test_profile = profile[test_indices]
    equipment_fault = ~np.all(
        np.isclose(test_profile[:, :4], IDEAL_PROFILE[:4]), axis=1
    )
    stability_issue = ~np.isclose(test_profile[:, 4], IDEAL_PROFILE[4])

    # Unstable cycles can differ from the healthy sensor reference even when
    # the four equipment-condition labels are optimal. Do not count these as
    # equipment-fault false alarms; assess them separately via stability_issue.
    evaluable = ~stability_issue
    predicted_fault = alerts != "Normal"
    y_pred = predicted_fault[evaluable]
    y_true = equipment_fault[evaluable]

    tp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))
    tn = int(np.sum(~y_pred & ~y_true))

    detection_rate = tp / (tp + fn) if (tp + fn) else np.nan
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    accuracy = (tp + tn) / int(evaluable.sum()) if evaluable.sum() else np.nan

    alert_df = pd.DataFrame({
        "cycle_id": test_indices + 1,
        "healthy_distance": healthy_distances[test_indices],
        "alert": alerts,
        "actual_equipment_fault": equipment_fault,
        "actual_stability_issue": stability_issue,
        "included_in_equipment_evaluation": evaluable,
    }).sort_values("cycle_id").reset_index(drop=True)

    evaluation_df = pd.DataFrame([
        {"metric": "TP", "value": tp},
        {"metric": "FP", "value": fp},
        {"metric": "FN", "value": fn},
        {"metric": "TN", "value": tn},
        {"metric": "Stable test cycles evaluated for equipment faults", "value": int(evaluable.sum())},
        {"metric": "Unstable test cycles excluded from equipment metrics", "value": int(stability_issue.sum())},
        {"metric": "Detection rate / Recall", "value": detection_rate},
        {"metric": "False alarm rate", "value": false_alarm_rate},
        {"metric": "Precision", "value": precision},
        {"metric": "Accuracy", "value": accuracy},
        {"metric": "Test cycles with stability issue", "value": int(stability_issue.sum())},
    ])

    return {
        "alerts": alert_df,
        "evaluation": evaluation_df,
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "precision": precision,
        "accuracy": accuracy,
        "stability_issue_count": int(stability_issue.sum()),
        "equipment_evaluation_count": int(evaluable.sum()),
    }


def run_analysis(dataset_dir=DATASET_DIR, threshold_95=None, threshold_99=None,
                 correlation_threshold=0.90):
    """Run all assignment analyses and return reusable results for UI and terminal."""
    data = load_dataset(dataset_dir)
    features = data["features"]
    profile = data["profile"]
    feature_names = data["feature_names"]
    sensor_data = data["sensor_data"]

    comparison = prepare_comparison(features, profile=profile)
    train_indices = comparison["train_indices"]
    test_indices = comparison["test_indices"]
    X_train_scaled = comparison["X_train_scaled"]
    X_test_scaled = comparison["X_test_scaled"]

    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df.insert(0, "cycle_id", np.arange(1, len(features) + 1))
    profile_df = pd.DataFrame(profile, columns=PROFILE_NAMES)
    profile_df.insert(0, "cycle_id", np.arange(1, len(profile) + 1))

    correlations, variable_mask = calculate_correlations(
        X_train_scaled, feature_names, threshold=correlation_threshold
    )
    independent = select_independent_features(X_train_scaled, feature_names)

    changes = calculate_cycle_changes(
        comparison["all_scaled"] if "all_scaled" in comparison
        else _assemble_all_scaled(comparison, len(features)),
        feature_names
    )
    

    healthy = calculate_healthy_distances(
        features, profile, train_indices, test_indices, X_train_scaled, X_test_scaled
    )
    thresholds = determine_thresholds(
        healthy, train_indices
    )
    final_95 = thresholds["threshold_95"] if threshold_95 is None else float(threshold_95)
    final_99 = thresholds["threshold_99"] if threshold_99 is None else float(threshold_99)
    if final_95 > final_99:
        raise ValueError("آستانه Normal باید کمتر یا مساوی آستانه Needs inspection باشد.")

    alert_evaluation = evaluate_alerts(
        profile, test_indices, healthy["healthy_distances"], final_95, final_99
    )

    # Independent-feature comparison: query the first test cycle against training rows.
    query_scaled = X_test_scaled[0]
    all_distances = np.linalg.norm(X_train_scaled - query_scaled, axis=1)
    indep_idx = independent["independent_indices"]
    independent_distances = np.linalg.norm(
        X_train_scaled[:, indep_idx] - query_scaled[indep_idx], axis=1
    )
    distance_comparison = pd.DataFrame({
        "cycle_id": train_indices + 1,
        "distance_all_features": all_distances,
        "distance_independent_features": independent_distances,
    }).sort_values("distance_all_features").reset_index(drop=True)

    # Verify a feature calculation on the first sensor and report the maxima.
    ps1 = sensor_data["PS1"]
    manual_cycle_index = 0
    manual_verification = {
        "cycle_id": 1,
        "mean": float(np.mean(ps1[manual_cycle_index])),
        "min": float(np.min(ps1[manual_cycle_index])),
        "max": float(np.max(ps1[manual_cycle_index])),
        "std": float(np.std(ps1[manual_cycle_index])),
        "rms": float(np.sqrt(np.mean(np.square(ps1[manual_cycle_index])))),
        "features_row": features[manual_cycle_index, :5].tolist(),
    }
    ts1_mean_index = feature_names.index("TS1_mean")
    vs1_rms_index = feature_names.index("VS1_rms")
    max_ts1_index = int(np.argmax(features[:, ts1_mean_index]))
    max_vs1_index = int(np.argmax(features[:, vs1_rms_index]))

    results = {
        **data,
        **comparison,
        "feature_df": feature_df,
        "profile_df": profile_df,
        "correlations": correlations,
        "variable_feature_mask": variable_mask,
        "independent": independent,
        "cycle_changes": changes,
        "healthy": healthy,
        "thresholds": thresholds,
        "threshold_95": final_95,
        "threshold_99": final_99,
        "alerts": alert_evaluation["alerts"],
        "evaluation": alert_evaluation["evaluation"],
        "evaluation_summary": alert_evaluation,
        "distance_comparison": distance_comparison,
        "manual_verification": manual_verification,
        "max_TS1_mean": {
            "cycle_id": max_ts1_index + 1,
            "value": float(features[max_ts1_index, ts1_mean_index]),
        },
        "max_VS1_rms": {
            "cycle_id": max_vs1_index + 1,
            "value": float(features[max_vs1_index, vs1_rms_index]),
        },
    }
    save_outputs(results)
    return results


def _assemble_all_scaled(comparison, number_of_cycles):
    """Place scaled train and test rows back in original cycle order."""
    all_scaled = np.empty((number_of_cycles, comparison["X_train_scaled"].shape[1]))
    all_scaled[comparison["train_indices"]] = comparison["X_train_scaled"]
    all_scaled[comparison["test_indices"]] = comparison["X_test_scaled"]
    return all_scaled


def save_outputs(results):
    """Save required analysis tables to an outputs folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results["feature_df"].to_csv(OUTPUT_DIR / "features.csv", index=False)
    results["standardization_params"].assign(
        feature_name=results["feature_names"]
    ).to_csv(OUTPUT_DIR / "standardization_params.csv", index=False)

    train_df = pd.DataFrame(results["X_train_scaled"], columns=results["feature_names"])
    train_df.insert(0, "cycle_id", results["train_indices"] + 1)
    train_df.to_csv(OUTPUT_DIR / "train_scaled.csv", index=False)

    test_df = pd.DataFrame(results["X_test_scaled"], columns=results["feature_names"])
    test_df.insert(0, "cycle_id", results["test_indices"] + 1)
    test_df.to_csv(OUTPUT_DIR / "test_scaled.csv", index=False)

    results["correlations"].to_csv(OUTPUT_DIR / "high_correlations.csv", index=False)
    results["cycle_changes"].to_csv(OUTPUT_DIR / "cycle_changes.csv", index=False)
    results["alerts"].to_csv(OUTPUT_DIR / "alerts.csv", index=False)
    results["evaluation"].to_csv(OUTPUT_DIR / "evaluation.csv", index=False)
    results["profile_df"].to_csv(OUTPUT_DIR / "profile_summary_source.csv", index=False)
    results["distance_comparison"].to_csv(OUTPUT_DIR / "distance_comparison.csv", index=False)

    healthy_df = pd.DataFrame({
        "cycle_id": np.arange(1, len(results["healthy"]["healthy_distances"]) + 1),
        "healthy_distance": results["healthy"]["healthy_distances"],
        "fully_healthy": results["healthy"]["fully_healthy_mask"],
    })
    healthy_df.to_csv(OUTPUT_DIR / "healthy_distances.csv", index=False)


def print_terminal_report(results):
    """Print a readable report; this function never requests terminal input."""
    print("\n" + "=" * 72)
    print("HYDRAULIC SYSTEM MONITORING — ANALYSIS REPORT")
    print("=" * 72)
    print(f"Cycles: {len(results['features']):,}")
    print(f"Sensors: {len(results['sensor_data'])}")
    print("Samples per cycle for each sensor:")
    for sensor, count in results["samples_per_cycle"].items():
       print(f"  {sensor}: {count}")
    print(f"Feature matrix: {results['features'].shape}")
    print(f"Profile matrix: {results['profile'].shape}")
    print(f"Train/test rows: {len(results['train_indices'])}/{len(results['test_indices'])}")
    print(f"Constant feature columns in training: {len(results['independent']['constant_columns'])}")
    print(f"Numerical matrix rank: {results['independent']['matrix_rank']}")
    print(f"Independent features retained: {len(results['independent']['independent_indices'])}")
    print(f"High-correlation pairs (|r| >= 0.90): {len(results['correlations'])}")
    print(f"Fully healthy training cycles: {results['healthy']['healthy_cycle_count']}")
    print(f"Threshold 95: {results['threshold_95']:.6f}")
    print(f"Threshold 99: {results['threshold_99']:.6f}")
    print(f"Threshold calibration samples: {results['thresholds']['calibration_sample_count']}")
    print(f"Threshold method: {results['thresholds']['calibration_method']}")
    print(f"Max TS1_mean: {results['max_TS1_mean']['value']:.6f} (cycle {results['max_TS1_mean']['cycle_id']})")
    print(f"Max VS1_rms: {results['max_VS1_rms']['value']:.6f} (cycle {results['max_VS1_rms']['cycle_id']})")
    print("\nConfusion matrix:")
    for key, value in results["evaluation_summary"]["confusion"].items():
        print(f"  {key}: {value}")
    print(f"Detection rate / Recall: {results['evaluation_summary']['detection_rate']:.4f}")
    print(f"False alarm rate: {results['evaluation_summary']['false_alarm_rate']:.4f}")
    print(f"Precision: {results['evaluation_summary']['precision']:.4f}")
    print(f"Accuracy: {results['evaluation_summary']['accuracy']:.4f}")
    print(f"Test cycles with stability issue: {results['evaluation_summary']['stability_issue_count']}")
    print(f"\nCSV outputs saved in: {OUTPUT_DIR}")
    
    if results["healthy"]["healthy_cycle_count"] < 30:
        print(
            "WARNING: Very few fully healthy training cycles were available. "
            "Distance thresholds are uncertain and should be interpreted as "
            "screening indicators, not definitive fault diagnoses."
        )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    # Terminal mode runs the complete analysis without asking for user input.
    report = run_analysis()
    print_terminal_report(report)
