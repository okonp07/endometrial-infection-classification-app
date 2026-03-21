from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a trained Keras model into the production app layout and write class names."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to the trained model file or directory.")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/endometrial_classifier.keras"),
        help="Destination path inside the production app.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Ordered class labels expected by the model.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("artifacts/class_names.json"),
        help="Path where the class-name JSON should be written.",
    )
    return parser.parse_args()


def copy_model(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Model path not found: {source}")

    if source.resolve() == destination.resolve():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    copy_model(args.model.resolve(), args.output_model.resolve())

    args.labels_path.parent.mkdir(parents=True, exist_ok=True)
    args.labels_path.write_text(json.dumps(args.labels, indent=2), encoding="utf-8")

    print(f"Model copied to {args.output_model.resolve()}")
    print(f"Class names written to {args.labels_path.resolve()}")


if __name__ == "__main__":
    main()
