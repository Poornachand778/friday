#!/usr/bin/env python3
"""
Friday Experiment Tracker
=========================

Track training experiments, compare results, and manage decisions.

Usage:
    python scripts/training/experiment_tracker.py new --name "Add backchannels"
    python scripts/training/experiment_tracker.py log EXP-001 --metrics eval_report.json
    python scripts/training/experiment_tracker.py compare EXP-001 EXP-002
    python scripts/training/experiment_tracker.py decide EXP-001 --keep
    python scripts/training/experiment_tracker.py list
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPERIMENTS_DIR = Path("data/training/experiments")
REGISTRY_PATH = EXPERIMENTS_DIR / "registry.json"


@dataclass
class Experiment:
    """A training experiment"""

    id: str
    name: str
    hypothesis: str
    created_at: str
    status: str = "planned"  # planned, running, completed, abandoned

    # Data configuration
    base_dataset: str = ""
    added_data: str = ""
    added_count: int = 0

    # Training config
    epochs: int = 3
    learning_rate: float = 2e-4
    lora_r: int = 16

    # Results
    training_loss: float = 0.0
    validation_loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    # Decision
    decision: str = ""  # keep, drop, modify
    decision_reason: str = ""
    next_steps: str = ""

    # Artifacts
    model_path: str = ""
    eval_report: str = ""
    notes: List[str] = field(default_factory=list)


class ExperimentTracker:
    """Manages training experiments"""

    def __init__(self):
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Experiment]:
        """Load experiment registry"""
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH) as f:
                data = json.load(f)
                return {
                    exp_id: Experiment(**exp_data) for exp_id, exp_data in data.items()
                }
        return {}

    def _save_registry(self):
        """Save experiment registry"""
        data = {exp_id: asdict(exp) for exp_id, exp in self.registry.items()}
        with open(REGISTRY_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_id(self) -> str:
        """Generate next experiment ID"""
        existing = [
            int(exp_id.split("-")[1])
            for exp_id in self.registry.keys()
            if exp_id.startswith("EXP-")
        ]
        next_num = max(existing, default=0) + 1
        return f"EXP-{next_num:03d}"

    def create_experiment(
        self,
        name: str,
        hypothesis: str = "",
        base_dataset: str = "",
        added_data: str = "",
    ) -> Experiment:
        """Create a new experiment"""
        exp_id = self._generate_id()

        exp = Experiment(
            id=exp_id,
            name=name,
            hypothesis=hypothesis,
            created_at=datetime.now().isoformat(),
            base_dataset=base_dataset,
            added_data=added_data,
        )

        self.registry[exp_id] = exp
        self._save_registry()

        # Create experiment directory
        exp_dir = EXPERIMENTS_DIR / exp_id
        exp_dir.mkdir(exist_ok=True)

        # Create experiment markdown file
        self._create_experiment_file(exp)

        print(f"Created experiment: {exp_id}")
        print(f"  Name: {name}")
        print(f"  Directory: {exp_dir}")

        return exp

    def _create_experiment_file(self, exp: Experiment):
        """Create markdown file for experiment"""
        exp_dir = EXPERIMENTS_DIR / exp.id
        exp_file = exp_dir / "README.md"

        content = f"""# {exp.id}: {exp.name}

**Created:** {exp.created_at}
**Status:** {exp.status}

## Hypothesis

{exp.hypothesis or "[Add your hypothesis here]"}

## Data Configuration

- **Base Dataset:** {exp.base_dataset or "[Specify base dataset]"}
- **Added Data:** {exp.added_data or "[Specify what data is added]"}
- **Added Count:** {exp.added_count}

## Training Configuration

- Epochs: {exp.epochs}
- Learning Rate: {exp.learning_rate}
- LoRA Rank: {exp.lora_r}

## Expected Outcomes

[What metrics do you expect to improve? By how much?]

## Results

| Metric | Baseline | This Experiment | Delta |
|--------|----------|-----------------|-------|
| Identity Score | | | |
| Brevity (tokens) | | | |
| Hedging Freq | | | |
| Flattery Freq | | | |
| Telugu Appropriate | | | |
| Boss Usage | | | |
| Overall | | | |

### Training Loss

- Final Training Loss: {exp.training_loss}
- Final Validation Loss: {exp.validation_loss}

### Observations

[Qualitative notes about the results]

## Decision

**Decision:** {exp.decision or "[KEEP / DROP / MODIFY]"}

**Reason:** {exp.decision_reason}

**Next Steps:** {exp.next_steps}

## Notes

{chr(10).join('- ' + note for note in exp.notes) if exp.notes else "[Add notes here]"}
"""

        with open(exp_file, "w") as f:
            f.write(content)

    def log_metrics(self, exp_id: str, metrics_file: str):
        """Log metrics from evaluation report"""
        if exp_id not in self.registry:
            print(f"Error: Experiment {exp_id} not found")
            return

        with open(metrics_file) as f:
            report = json.load(f)

        exp = self.registry[exp_id]
        exp.metrics = report.get("metrics", {})
        exp.status = "completed"
        exp.eval_report = metrics_file

        self._save_registry()
        self._create_experiment_file(exp)

        print(f"Logged metrics for {exp_id}")
        for metric, value in exp.metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")

    def update_training_stats(
        self,
        exp_id: str,
        training_loss: float,
        validation_loss: float,
        model_path: str = "",
    ):
        """Update training loss values"""
        if exp_id not in self.registry:
            print(f"Error: Experiment {exp_id} not found")
            return

        exp = self.registry[exp_id]
        exp.training_loss = training_loss
        exp.validation_loss = validation_loss
        exp.model_path = model_path
        exp.status = "completed"

        self._save_registry()
        self._create_experiment_file(exp)

        print(f"Updated training stats for {exp_id}")

    def decide(
        self, exp_id: str, decision: str, reason: str = "", next_steps: str = ""
    ):
        """Record decision for experiment"""
        if exp_id not in self.registry:
            print(f"Error: Experiment {exp_id} not found")
            return

        if decision not in ["keep", "drop", "modify"]:
            print(f"Error: Decision must be 'keep', 'drop', or 'modify'")
            return

        exp = self.registry[exp_id]
        exp.decision = decision
        exp.decision_reason = reason
        exp.next_steps = next_steps

        self._save_registry()
        self._create_experiment_file(exp)

        print(f"Decision recorded for {exp_id}: {decision.upper()}")

    def add_note(self, exp_id: str, note: str):
        """Add a note to experiment"""
        if exp_id not in self.registry:
            print(f"Error: Experiment {exp_id} not found")
            return

        exp = self.registry[exp_id]
        exp.notes.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}")

        self._save_registry()
        self._create_experiment_file(exp)

        print(f"Note added to {exp_id}")

    def compare(self, exp_id_1: str, exp_id_2: str):
        """Compare two experiments"""
        if exp_id_1 not in self.registry or exp_id_2 not in self.registry:
            print("Error: One or both experiments not found")
            return

        exp1 = self.registry[exp_id_1]
        exp2 = self.registry[exp_id_2]

        print(f"\n{'='*60}")
        print(f"COMPARISON: {exp_id_1} vs {exp_id_2}")
        print(f"{'='*60}")
        print(f"\n{exp_id_1}: {exp1.name}")
        print(f"{exp_id_2}: {exp2.name}")

        # Compare metrics
        all_metrics = set(exp1.metrics.keys()) | set(exp2.metrics.keys())

        print(f"\n{'Metric':<25} {exp_id_1:<12} {exp_id_2:<12} {'Delta':<10}")
        print("-" * 60)

        for metric in sorted(all_metrics):
            v1 = exp1.metrics.get(metric, 0)
            v2 = exp2.metrics.get(metric, 0)
            delta = v2 - v1

            # Color-code delta
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                delta_str = f"{delta:+.3f}"
                if "hedging" in metric or "flattery" in metric or "brevity" in metric:
                    # Lower is better for these
                    better = delta < 0
                else:
                    better = delta > 0

                indicator = "↑" if better else "↓" if delta != 0 else "="
                print(
                    f"{metric:<25} {v1:<12.3f} {v2:<12.3f} {delta_str:<10} {indicator}"
                )
            else:
                print(f"{metric:<25} {v1!s:<12} {v2!s:<12}")

        print("-" * 60)

        # Overall verdict
        overall_1 = exp1.metrics.get("overall_score", 0)
        overall_2 = exp2.metrics.get("overall_score", 0)

        if overall_2 > overall_1:
            print(
                f"\n✓ {exp_id_2} is better overall ({overall_2:.3f} vs {overall_1:.3f})"
            )
        elif overall_1 > overall_2:
            print(
                f"\n✓ {exp_id_1} is better overall ({overall_1:.3f} vs {overall_2:.3f})"
            )
        else:
            print(f"\n= Both experiments have similar overall scores")

    def list_experiments(self):
        """List all experiments"""
        if not self.registry:
            print("No experiments found")
            return

        print(f"\n{'='*70}")
        print("EXPERIMENTS")
        print(f"{'='*70}")
        print(f"{'ID':<10} {'Status':<12} {'Decision':<10} {'Name':<35}")
        print("-" * 70)

        for exp_id, exp in sorted(self.registry.items()):
            status = exp.status
            decision = exp.decision or "-"
            name = exp.name[:35]

            # Status indicator
            if status == "completed":
                status_icon = "✓"
            elif status == "running":
                status_icon = "⚙"
            else:
                status_icon = "○"

            # Decision indicator
            if decision == "keep":
                decision_icon = "✓"
            elif decision == "drop":
                decision_icon = "✗"
            elif decision == "modify":
                decision_icon = "~"
            else:
                decision_icon = " "

            print(
                f"{exp_id:<10} {status_icon} {status:<10} {decision_icon} {decision:<8} {name}"
            )

        print("-" * 70)
        print(f"Total: {len(self.registry)} experiments")

    def show_experiment(self, exp_id: str):
        """Show detailed experiment info"""
        if exp_id not in self.registry:
            print(f"Error: Experiment {exp_id} not found")
            return

        exp = self.registry[exp_id]

        print(f"\n{'='*60}")
        print(f"{exp.id}: {exp.name}")
        print(f"{'='*60}")
        print(f"\nStatus: {exp.status}")
        print(f"Created: {exp.created_at}")

        if exp.hypothesis:
            print(f"\nHypothesis:")
            print(f"  {exp.hypothesis}")

        print(f"\nData:")
        print(f"  Base: {exp.base_dataset}")
        print(f"  Added: {exp.added_data} ({exp.added_count} examples)")

        if exp.metrics:
            print(f"\nMetrics:")
            for metric, value in exp.metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.3f}")

        if exp.decision:
            print(f"\nDecision: {exp.decision.upper()}")
            if exp.decision_reason:
                print(f"  Reason: {exp.decision_reason}")
            if exp.next_steps:
                print(f"  Next: {exp.next_steps}")

        if exp.notes:
            print(f"\nNotes:")
            for note in exp.notes[-5:]:  # Last 5 notes
                print(f"  {note}")


def main():
    parser = argparse.ArgumentParser(description="Friday Experiment Tracker")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # New experiment
    new_parser = subparsers.add_parser("new", help="Create new experiment")
    new_parser.add_argument("--name", required=True, help="Experiment name")
    new_parser.add_argument("--hypothesis", default="", help="Hypothesis")
    new_parser.add_argument("--base", default="", help="Base dataset")
    new_parser.add_argument("--added", default="", help="Added data description")

    # Log metrics
    log_parser = subparsers.add_parser("log", help="Log metrics from eval report")
    log_parser.add_argument("exp_id", help="Experiment ID")
    log_parser.add_argument("--metrics", required=True, help="Path to eval report JSON")

    # Update training stats
    train_parser = subparsers.add_parser("train", help="Update training stats")
    train_parser.add_argument("exp_id", help="Experiment ID")
    train_parser.add_argument("--train-loss", type=float, required=True)
    train_parser.add_argument("--val-loss", type=float, required=True)
    train_parser.add_argument("--model", default="", help="Model path")

    # Decide
    decide_parser = subparsers.add_parser("decide", help="Record decision")
    decide_parser.add_argument("exp_id", help="Experiment ID")
    decide_parser.add_argument("--keep", action="store_true")
    decide_parser.add_argument("--drop", action="store_true")
    decide_parser.add_argument("--modify", action="store_true")
    decide_parser.add_argument("--reason", default="", help="Decision reason")
    decide_parser.add_argument("--next", default="", help="Next steps")

    # Note
    note_parser = subparsers.add_parser("note", help="Add note")
    note_parser.add_argument("exp_id", help="Experiment ID")
    note_parser.add_argument("--text", required=True, help="Note text")

    # Compare
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("exp1", help="First experiment ID")
    compare_parser.add_argument("exp2", help="Second experiment ID")

    # List
    subparsers.add_parser("list", help="List all experiments")

    # Show
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("exp_id", help="Experiment ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    tracker = ExperimentTracker()

    if args.command == "new":
        tracker.create_experiment(
            name=args.name,
            hypothesis=args.hypothesis,
            base_dataset=args.base,
            added_data=args.added,
        )

    elif args.command == "log":
        tracker.log_metrics(args.exp_id, args.metrics)

    elif args.command == "train":
        tracker.update_training_stats(
            args.exp_id,
            args.train_loss,
            args.val_loss,
            args.model,
        )

    elif args.command == "decide":
        if args.keep:
            decision = "keep"
        elif args.drop:
            decision = "drop"
        elif args.modify:
            decision = "modify"
        else:
            print("Error: Specify --keep, --drop, or --modify")
            return
        tracker.decide(args.exp_id, decision, args.reason, getattr(args, "next", ""))

    elif args.command == "note":
        tracker.add_note(args.exp_id, args.text)

    elif args.command == "compare":
        tracker.compare(args.exp1, args.exp2)

    elif args.command == "list":
        tracker.list_experiments()

    elif args.command == "show":
        tracker.show_experiment(args.exp_id)


if __name__ == "__main__":
    main()
