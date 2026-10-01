#!/usr/bin/env python3
"""
Benchmark the voice-detection models on labelled audio clips.

The data folder must contain two subfolders:
    real/    human recordings   (label 0)
    spoof/   AI-generated clips (label 1)

Paths are resolved from the repo root, so this can be run from any folder.

Examples:
    python benchmarks/benchmark.py                                  # deployed model, tests/samples
    python benchmarks/benchmark.py --models general tts elevenlabs  # compare models
    python benchmarks/benchmark.py --models general int8 xgb        # Keras vs TFLite vs XGBoost
    python benchmarks/benchmark.py --ensemble general tts           # ensemble strategies
    python benchmarks/benchmark.py --data /path/to/clips --sample-percent 5
    python benchmarks/benchmark.py --min-accuracy 80 --max-latency-ms 50   # exit 1 if missed (for CI)

Exit codes: 0 = OK, 1 = a threshold was missed, 2 = no data or primary model missing.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Feature settings - must match app.py
SAMPLE_RATE = 16000
N_LFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
TARGET_TIME_STEPS = 312
MAX_DURATION = 10.0
THRESHOLD = 0.5  # score >= THRESHOLD -> AI_GENERATED

# Short name -> (path from repo root, model type, description)
MODELS = {
    "general":    ("model/best_model_general.h5",                    "keras",   "Keras CNN, general (deployed)"),
    "tts":        ("model/best_model_tts.h5",                        "keras",   "Keras CNN, TTS"),
    "elevenlabs": ("model/best_model_elevenlabs.h5",                 "keras",   "Keras CNN, ElevenLabs"),
    "int8":       ("model/model_int8.tflite",                        "tflite",  "TFLite INT8 (ElevenLabs CNN)"),
    "xgb":        ("model/model_xgboost.json",                       "xgboost", "XGBoost"),
    "xgb-depth4": ("model/model_xgboost_depth4.json",                "xgboost", "XGBoost, depth 4"),
    "xgb-ubj":    ("model/model_xgboost.ubj",                        "xgboost", "XGBoost, UBJ format"),
}
NORM_PARAMS = REPO_ROOT / "model" / "normalization_params.npz"
AUDIO_EXTENSIONS = ("*.mp3", "*.wav", "*.flac", "*.m4a")
LANGUAGES = {"en": "English", "ma": "Malayalam", "te": "Telugu", "ta": "Tamil", "hi": "Hindi"}


# ============== Data ==============
@dataclass
class Sample:
    path: str
    label: int       # 0 = human, 1 = AI
    language: str
    cnn: np.ndarray  # normalized, shape (1, 40, 312, 1) - Keras / TFLite input
    raw: np.ndarray  # unnormalized, shape (40, 312) - XGBoost input


def language_of(path):
    """Language code from the filename prefix, e.g. 'Tamil_...' -> 'ta'."""
    name = os.path.basename(path).lower()
    for code in LANGUAGES:
        if name.startswith(code):
            return code
    return "unknown"


def extract_features(path, mean, std):
    """Load audio once and return (normalized CNN input, raw LFCC matrix)."""
    import librosa

    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=MAX_DURATION)
    lfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_LFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    if lfcc.shape[1] < TARGET_TIME_STEPS:
        lfcc = np.pad(lfcc, ((0, 0), (0, TARGET_TIME_STEPS - lfcc.shape[1])), mode="constant")
    else:
        lfcc = lfcc[:, :TARGET_TIME_STEPS]
    raw = lfcc.astype(np.float32)
    cnn = ((raw - mean) / (std + 1e-8))[np.newaxis, ..., np.newaxis].astype(np.float32)
    return cnn, raw


def find_audio(folder):
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(Path(folder).rglob(ext))
    return sorted(files)


def load_dataset(data_dir, sample_percent, seed):
    norm = np.load(NORM_PARAMS)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    if mean.ndim == 1:
        mean, std = mean[:, np.newaxis], std[:, np.newaxis]

    rng = random.Random(seed)
    dataset = []
    for sub, label in (("real", 0), ("spoof", 1)):
        files = find_audio(Path(data_dir) / sub)
        if sample_percent < 100 and files:
            files = rng.sample(files, max(1, int(len(files) * sample_percent / 100)))
        print(f"  {sub}/: using {len(files)} files")
        for f in files:
            try:
                cnn, raw = extract_features(str(f), mean, std)
                dataset.append(Sample(str(f), label, language_of(str(f)), cnn, raw))
            except Exception as e:
                print(f"  ! skipped {f.name}: {e}")
    return dataset


# ============== Model wrappers ==============
def build_cnn_model(input_shape=(N_LFCC, TARGET_TIME_STEPS, 1)):
    """Same architecture as app.py; used if an .h5 file only holds weights."""
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=input_shape, name="input")

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="dropout1")(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="dropout2")(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(0.25, name="dropout3")(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.Activation("relu", name="relu4")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Dense layers
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.5, name="dropout4")(x)
    x = layers.Dense(64, activation="relu", name="dense2")(x)
    x = layers.Dropout(0.5, name="dropout5")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name="VoiceClassifierCNN")


class KerasModel:
    def __init__(self, path):
        import tensorflow as tf

        try:
            self.model = tf.keras.models.load_model(path, compile=False)
        except Exception:
            self.model = build_cnn_model()
            self.model.load_weights(path)
        self.model.trainable = False

    def predict(self, sample):
        return float(self.model(sample.cnn, training=False)[0][0].numpy())


class TFLiteModel:
    def __init__(self, path):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def predict(self, sample):
        x = sample.cnn
        if self.inp["dtype"] == np.uint8:
            scale, zero_point = self.inp["quantization"]
            x = np.nan_to_num(x / scale + zero_point, nan=0.0, posinf=255.0, neginf=0.0)
            x = np.clip(x, 0, 255).astype(np.uint8)
        self.interpreter.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])
        if self.out["dtype"] == np.uint8:
            scale, zero_point = self.out["quantization"]
            y = (y.astype(np.float32) - zero_point) * scale
        return float(y[0][0])


class XGBoostModel:
    def __init__(self, path):
        import xgboost as xgb

        self.xgb = xgb
        self.model = xgb.Booster()
        self.model.load_model(path)

    @staticmethod
    def statistical_features(m):
        """(40, 312) raw LFCC -> 365 summary features, as used in training."""
        p25, p50, p75 = (np.percentile(m, q, axis=1) for q in (25, 50, 75))
        mx, mn = np.max(m, axis=1), np.min(m, axis=1)
        per_coef = [np.mean(m, axis=1), np.std(m, axis=1), mn, mx, p25, p50, p75, mx - mn, p75 - p25]
        overall = [np.mean(m), np.std(m), np.min(m), np.max(m), np.median(m)]
        return np.concatenate(per_coef + [np.array(overall)]).reshape(1, -1).astype(np.float32)

    def predict(self, sample):
        dmatrix = self.xgb.DMatrix(self.statistical_features(sample.raw))
        return float(self.model.predict(dmatrix)[0])


LOADERS = {"keras": KerasModel, "tflite": TFLiteModel, "xgboost": XGBoostModel}


def resolve_model(name):
    """Accept a short name from MODELS or a file path."""
    if name in MODELS:
        rel, kind, desc = MODELS[name]
        return name, REPO_ROOT / rel, kind, desc
    path = Path(name)
    kind = {".h5": "keras", ".keras": "keras", ".tflite": "tflite",
            ".json": "xgboost", ".ubj": "xgboost"}.get(path.suffix)
    if kind is None:
        raise ValueError(f"Unknown model '{name}'. Use one of: {', '.join(MODELS)} or a file path.")
    return path.stem, path, kind, path.name


# ============== Running and scoring ==============
def run_model(model, dataset):
    """Return per-sample scores, latencies (ms) and CPU readings."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except ImportError:
        proc = None
        print("  (psutil not installed: CPU usage not measured)")

    scores, times_ms, cpu = [], [], []
    for s in dataset:
        model.predict(s)  # warm-up, not timed
        if proc:
            proc.cpu_percent()
        start = time.perf_counter()
        scores.append(model.predict(s))
        times_ms.append((time.perf_counter() - start) * 1000)
        if proc:
            cpu.append(proc.cpu_percent())
    return scores, times_ms, cpu


def summarize(name, scores, times_ms, cpu, dataset):
    preds = [1 if p >= THRESHOLD else 0 for p in scores]
    errors, lang_stats = [], {}
    for s, p, pred in zip(dataset, scores, preds):
        stats = lang_stats.setdefault(s.language, {"total": 0, "errors": 0})
        stats["total"] += 1
        if pred != s.label:
            stats["errors"] += 1
            errors.append({
                "file": os.path.basename(s.path),
                "actual": "AI" if s.label else "Human",
                "predicted": "AI" if pred else "Human",
                "confidence": p if pred else 1 - p,
                "raw_prediction": p,
                "language": s.language,
            })
    correct = len(preds) - len(errors)
    t = np.array(times_ms)
    return {
        "model_name": name,
        "accuracy": correct / len(preds) * 100,
        "avg_inference_ms": float(t.mean()),
        "std_inference_ms": float(t.std()),
        "min_inference_ms": float(t.min()),
        "max_inference_ms": float(t.max()),
        "p95_inference_ms": float(np.percentile(t, 95)),
        "avg_cpu_percent": float(np.mean(cpu)) if cpu else None,
        "total_samples": len(preds),
        "correct_predictions": correct,
        "total_errors": len(errors),
        "errors_by_language": {k: v["errors"] for k, v in lang_stats.items()},
        "language_stats": lang_stats,
        "error_details": errors[:20],
    }


def print_result(r):
    cpu = f"{r['avg_cpu_percent']:.1f}%" if r["avg_cpu_percent"] is not None else "n/a"
    print(f"  Accuracy: {r['accuracy']:.2f}%  ({r['correct_predictions']}/{r['total_samples']})")
    print(f"  Latency:  avg {r['avg_inference_ms']:.2f} ms | p95 {r['p95_inference_ms']:.2f} ms | "
          f"min {r['min_inference_ms']:.2f} | max {r['max_inference_ms']:.2f}")
    print(f"  CPU:      {cpu}")
    for code, st in sorted(r["language_stats"].items()):
        print(f"    {LANGUAGES.get(code, 'Unknown'):<10} {st['errors']}/{st['total']} errors")


def ensemble_results(names, runs, dataset, weights):
    """Combine stored per-sample scores. Latency = sum of member latencies (run one after another)."""
    member_scores = [np.array(runs[n][0]) for n in names]
    member_times = np.sum([runs[n][1] for n in names], axis=0)
    label = "+".join(names)

    configs = [("average", None, f"Ensemble {label} (average)")]
    if len(names) == 2:
        for w in weights:
            if 0 < w < 1 and abs(w - 0.5) > 1e-9:  # 0.5 is the same as average
                configs.append(("weighted", [w, 1 - w],
                                f"Ensemble {label} ({round(w * 100)}/{round((1 - w) * 100)})"))

    results = []
    for strategy, w, name in configs:
        if w is None:
            combined = np.mean(member_scores, axis=0)
        else:
            combined = w[0] * member_scores[0] + w[1] * member_scores[1]
        r = summarize(name, combined.tolist(), member_times.tolist(), [], dataset)
        r.update({"strategy": strategy, "members": names, "weights": w})
        r["detailed_predictions"] = [
            {"file": os.path.basename(s.path), "actual_label": s.label, "language": s.language,
             **{f"{n}_pred": float(runs[n][0][i]) for n in names}, "ensemble_pred": float(combined[i])}
            for i, s in enumerate(dataset)
        ][:50]
        results.append(r)
    return results


# ============== Main ==============
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["general"],
                   help=f"models to benchmark; the first is the 'primary' one checked against thresholds. "
                        f"Names: {', '.join(MODELS)}, or a file path (default: general)")
    p.add_argument("--ensemble", nargs="+", metavar="MODEL",
                   help="also benchmark an ensemble of these models (Keras/TFLite/XGBoost can be mixed)")
    p.add_argument("--weights", nargs="+", type=float, default=[0.6, 0.4],
                   help="for a 2-model ensemble: weight given to the first model (default: 0.6 0.4)")
    p.add_argument("--data", default=str(REPO_ROOT / "tests" / "samples"),
                   help="folder containing real/ and spoof/ (default: tests/samples)")
    p.add_argument("--sample-percent", type=float, default=100, help="use a random subset (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="random seed for --sample-percent (default: 42)")
    p.add_argument("--out", help="results JSON path (default: benchmarks/results/benchmark_<time>.json)")
    p.add_argument("--min-accuracy", type=float, help="exit 1 if the primary model's accuracy (%%) is below this")
    p.add_argument("--max-latency-ms", type=float, help="exit 1 if the primary model's p95 latency is above this")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        for name in args.models + (args.ensemble or []):
            resolve_model(name)
    except ValueError as e:
        print(e)
        return 2

    print(f"Loading data from {args.data}")
    dataset = load_dataset(args.data, args.sample_percent, args.seed)
    if not dataset:
        print("No audio found. Expected real/ and spoof/ subfolders.")
        return 2

    wanted = list(dict.fromkeys(args.models + (args.ensemble or [])))  # unique, keep order
    runs, results = {}, []
    for name in wanted:
        short, path, kind, desc = resolve_model(name)
        if not path.exists():
            print(f"\n! Skipping {short}: file not found ({path})")
            if name == args.models[0]:
                return 2
            continue
        print(f"\n=== {desc} [{short}] ===")
        try:
            model = LOADERS[kind](str(path))
        except Exception as e:
            print(f"! Failed to load {short}: {e}")
            if name == args.models[0]:
                return 2
            continue
        runs[name] = run_model(model, dataset)
        if name in args.models:
            r = summarize(desc, *runs[name], dataset)
            r.update({"model_key": short, "model_path": str(path),
                      "model_size_mb": path.stat().st_size / 1024 / 1024})
            print_result(r)
            results.append(r)

    if args.ensemble:
        members = [n for n in args.ensemble if n in runs]
        if len(members) < 2:
            print("\n! Ensemble needs at least 2 models that loaded; skipping.")
        else:
            for r in ensemble_results(members, runs, dataset, args.weights):
                print(f"\n=== {r['model_name']} ===")
                print_result(r)
                results.append(r)

    # Summary table
    print(f"\n{'Model':<42} {'Size MB':>8} {'Accuracy':>9} {'Avg ms':>8} {'p95 ms':>8} {'Errors':>7}")
    print("-" * 87)
    for r in results:
        size = f"{r['model_size_mb']:.2f}" if "model_size_mb" in r else "-"
        print(f"{r['model_name']:<42} {size:>8} {r['accuracy']:>8.2f}% "
              f"{r['avg_inference_ms']:>8.2f} {r['p95_inference_ms']:>8.2f} {r['total_errors']:>7}")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "results" / f"benchmark_{datetime.now():%Y%m%d-%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "data_dir": args.data,
        "dataset_size": len(dataset),
        "test_percent": args.sample_percent,
        "threshold": THRESHOLD,
        "results": results,
    }, indent=2))
    print(f"\nResults saved to {out}")

    # Thresholds (for CI): checked against the primary model only
    primary = next((r for r in results if r.get("model_key") == resolve_model(args.models[0])[0]), None)
    failed = []
    if primary and args.min_accuracy is not None and primary["accuracy"] < args.min_accuracy:
        failed.append(f"accuracy {primary['accuracy']:.2f}% < {args.min_accuracy}%")
    if primary and args.max_latency_ms is not None and primary["p95_inference_ms"] > args.max_latency_ms:
        failed.append(f"p95 latency {primary['p95_inference_ms']:.2f} ms > {args.max_latency_ms} ms")
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
