"""Run a prediction on a local PNG image."""

import argparse
import json
from pathlib import Path

from .inference import load_model, predict_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.keras"))
    args = parser.parse_args()
    print(json.dumps({"predictions": predict_bytes(
        load_model(args.model), args.image.read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
