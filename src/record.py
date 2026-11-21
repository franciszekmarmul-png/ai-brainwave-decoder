from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.live import MODELS, RunRequest, run
from src.server import SOURCE_FILES, options

ROOT = Path(__file__).resolve().parents[1]


def _rounded(event: dict) -> dict:
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in event.items()}


def record(datasets: list[str], people: list[int], time_limit: float) -> dict:
    runs = []
    for dataset in datasets:
        for person in people:
            for model in MODELS:
                events: list[dict] = []
                request = RunRequest(dataset, person, model, time_limit)
                summary = run(request, lambda event: events.append(_rounded(event)))
                print(f"{dataset} person {person} {model}: best {summary['best_accuracy']:.1%} at {summary['best_time']:.1f} s")
                runs.append({"dataset": dataset, "person": person, "model": model, "time_limit": time_limit, "events": events})
    catalogue = options()
    for entry in catalogue["datasets"]:
        entry["downloaded"] = True
    sources = {name: (ROOT / "src" / name).read_text() for name in SOURCE_FILES}
    return {"options": catalogue, "sources": sources, "runs": runs}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record real training runs for the hosted replay page.")
    parser.add_argument("--datasets", nargs="+", default=["bnci2014_001", "bnci2014_004", "physionet_mi"])
    parser.add_argument("--people", nargs="+", type=int, default=[1])
    parser.add_argument("--time-limit", type=float, default=120)
    parser.add_argument("--out", type=Path, default=ROOT / "web" / "replay" / "runs.json")
    args = parser.parse_args(argv)
    recording = record(args.datasets, args.people, args.time_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recording, separators=(",", ":")))
    print(f"Saved {len(recording['runs'])} runs to {args.out}")


if __name__ == "__main__":
    main()
