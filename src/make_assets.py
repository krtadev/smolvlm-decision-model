"""Build the plots, the examples page and the animation from finished SmolVLM Phase 1 runs (CPU only, no torch)."""
import base64
import io
import json
import os
import random
import shutil

import matplotlib
import numpy as np
from datasets import load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))           # src/
ROOT = os.path.dirname(HERE)                                # repo root: index.html and examples.html (GitHub Pages)
RUNS_DIR = os.path.join(ROOT, "runs")                       # one folder per Colab run, as downloaded (not committed)
TEMPLATES = os.path.join(HERE, "templates")
RESULTS = os.path.join(ROOT, "results")                     # chosen run's numbers, logits and log
FIGURES = os.path.join(RESULTS, "figures")
MODEL = os.path.join(ROOT, "model")                         # chosen run's LoRA adapter
EVAL_SEED = 0               # option order of the saved test logits (seed 0 in phase1_smolvlm.py)
N_ANIMATION = 48            # questions the animation answers one by one
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
LIGHT_BLUE, REF_GREY = "#86b6ef", "#a3a29d"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs():
    # Every finished run (results.json with a comparison block) under RUNS_DIR, keyed by folder name.
    runs = {}
    for name in sorted(os.listdir(RUNS_DIR)):
        path = os.path.join(RUNS_DIR, name, "results.json")
        if os.path.exists(path):
            results = json.load(open(path))
            if "comparison" in results:
                runs[name] = results
    return runs


def pick_run(runs):
    # The run whose best checkpoint has the lowest validation-subset loss (chosen on validation, never on test).
    return min(runs, key=lambda n: min(v["val_loss"] for v in runs[n]["training"]["val_history"]))


def run_label(results):
    # Short human name for a run's settings.
    c = results["config"]
    return f"LoRA rank {c['lora_rank']}, lr {c['lr']:g}"


def shuffle_choices(row, rng):
    # Return a copy of the row with choices shuffled and the answer index updated (same as phase1_smolvlm.py).
    order = list(range(len(row["choices"])))
    rng.shuffle(order)
    return {**row, "choices": [row["choices"][i] for i in order], "answer": order.index(row["answer"])}


def load_test_rows():
    # Test rows with images, options in the same order the saved logits were scored in.
    ds = load_dataset("derek-thomas/ScienceQA", split="test")
    rows = [r for r in ds if r["image"] is not None]
    return [shuffle_choices(r, random.Random(EVAL_SEED * 100003 + i)) for i, r in enumerate(rows)]


def softmax(logits, T):
    # Row-wise softmax of logits / T.
    z = logits / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def style_axes(ax):
    # Recessive grid, no top/right spines, muted tick text.
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def trailing_mean(values, window=5):
    # Mean of each value and the (window - 1) before it.
    return [float(np.mean(values[max(0, i - window + 1):i + 1])) for i in range(len(values))]


def plot_loss_curve(results, path):
    # Training loss (20-step means, plus a 5-point trailing average) with validation-subset loss at each check.
    t = results["training"]
    steps, losses = [p["step"] for p in t["losses"]], [p["loss"] for p in t["losses"]]
    val = t["val_history"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    style_axes(ax)
    ax.plot(steps, losses, color=BLUE, alpha=0.3, linewidth=1)
    ax.plot(steps, trailing_mean(losses), color=BLUE, linewidth=2)
    ax.plot([v["step"] for v in val], [v["val_loss"] for v in val], color=ORANGE, linewidth=2, marker="o", markersize=7)
    ax.axvline(t["best_step"], color=MUTED, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("training step (batch of 16)", color=MUTED)
    ax.set_ylabel("cross-entropy loss (log scale)", color=MUTED)
    ax.text(steps[-1], trailing_mean(losses)[-1] * 0.93, "training loss\n(5-point average of 20-step means)", color=INK, ha="right", va="top", fontsize=9)
    ax.text(val[0]["step"] + 20, val[0]["val_loss"] * 1.15, "validation-subset loss (500 rows)", color=INK, fontsize=9)
    ax.text(t["best_step"] + 15, ax.get_ylim()[0] * 1.1, f"best checkpoint: step {t['best_step']}", color=MUTED, fontsize=9)
    ax.set_title(f"SmolVLM-256M: loss vs training step ({run_label(results)})", color=INK, loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reliability(results, path):
    # Reliability diagram on the test set: confidence vs accuracy per bin for the three models.
    fig, ax = plt.subplots(figsize=(6, 6))
    style_axes(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    for key, name, color in (("zero_shot", "zero-shot", BLUE), ("fine_tuned", "fine-tuned", ORANGE), ("fine_tuned_calibrated", "fine-tuned + calibrated", AQUA)):
        bins = [b for b in results[key]["main"]["reliability"] if b["count"] > 0]
        ece = results[key]["main"]["ece"]
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], color=color, linewidth=2, marker="o", markersize=8,
                label=f"{name} (ECE {ece:.3f})")
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("mean confidence in bin", color=MUTED)
    ax.set_ylabel("accuracy in bin", color=MUTED)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("SmolVLM-256M: reliability on the test set (10 bins)", color=INK, loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def accuracy_rows(runs, best):
    # Test accuracy rows, top to bottom: random guess, zero-shot, then every fine-tuned run (the chosen one marked).
    c = runs[best]["comparison"]
    rows = [{"label": "random guess", "value": c["zero_shot"]["chance_accuracy"], "kind": "ref"},
            {"label": "zero-shot, image blanked", "value": c["zero_shot"]["accuracy_without_image"], "kind": "dim"},
            {"label": "zero-shot", "value": c["zero_shot"]["accuracy"], "ci": c["zero_shot"]["accuracy_ci95"], "kind": "main"},
            {"label": "fine-tuned, image blanked", "value": c["fine_tuned"]["accuracy_without_image"], "kind": "dim"}]
    for name, results in runs.items():
        ft = results["comparison"]["fine_tuned"]
        rows.append({"label": f"fine-tuned, {run_label(results)}" + (" (chosen)" if name == best and len(runs) > 1 else ""),
                     "value": ft["accuracy"], "ci": ft["accuracy_ci95"], "kind": "main" if name == best else "dim"})
    return rows


def plot_accuracy(rows, path):
    # Horizontal bars of test accuracy with 95% intervals and the value written at each bar end.
    colors = {"main": BLUE, "dim": LIGHT_BLUE, "ref": REF_GREY}
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.2))
    style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["value"] for r in rows], height=0.62, color=[colors[r["kind"]] for r in rows],
            xerr=[r.get("ci", 0) for r in rows], error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    for yi, r in zip(y, rows):
        ci = f" ± {100 * r['ci']:.1f}" if "ci" in r else ""
        ax.text(r["value"] + r.get("ci", 0) + 0.01, yi, f"{100 * r['value']:.1f}%{ci}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("ScienceQA test accuracy (2,017 questions with images)", color=INK, loc="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def image_to_b64(image, side=448):
    # Shrink and JPEG-encode a PIL image as a base64 string for embedding in HTML.
    image = image.convert("RGB")
    image.thumbnail((side, side))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def make_example(index, row, probs):
    # One example: image, question, options, calibrated probabilities and the option the model chose.
    n = len(row["choices"])
    return {"test_index": index, "image": image_to_b64(row["image"]), "question": row["question"], "hint": row["hint"],
            "subject": row["subject"], "grade": row["grade"], "choices": row["choices"], "answer": row["answer"],
            "probabilities": [float(p) for p in probs[:n]], "choice": int(probs[:n].argmax())}


def write_page(template, data, path):
    # Fill a template's __DATA__ slot with JSON and write the self-contained page.
    page = open(os.path.join(TEMPLATES, template), encoding="utf-8").read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(page.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))


def copy_chosen_run(run_dir):
    # Copy the chosen run's adapter into model/ and its results, logits and log into results/.
    shutil.copytree(os.path.join(run_dir, "adapter"), os.path.join(MODEL, "adapter"), dirs_exist_ok=True)
    for name in ("results.json", "logits.npz", "run.log"):
        shutil.copy(os.path.join(run_dir, name), os.path.join(RESULTS, name))


def main():
    # Pick the best run on validation, then build every asset from it.
    runs = load_runs()
    best = pick_run(runs)
    results = runs[best]
    print("runs:", list(runs), "-> chosen on validation loss:", best)
    os.makedirs(FIGURES, exist_ok=True)
    copy_chosen_run(os.path.join(RUNS_DIR, best))

    plot_loss_curve(results, os.path.join(FIGURES, "loss_curve.png"))
    plot_reliability(results, os.path.join(FIGURES, "reliability.png"))
    rows = accuracy_rows(runs, best)
    plot_accuracy(rows, os.path.join(FIGURES, "accuracy.png"))
    json.dump({"chosen": best, "rows": rows}, open(os.path.join(RESULTS, "runs_summary.json"), "w"), indent=2)

    test = load_test_rows()
    saved = np.load(os.path.join(RUNS_DIR, best, "logits.npz"))
    assert (np.array([r["answer"] for r in test]) == saved["test_labels"]).all(), "row order does not match saved logits"
    probs = softmax(saved["test"], results["temperature"])
    correct = probs.argmax(1) == saved["test_labels"]
    right, wrong = list(np.flatnonzero(correct)), list(np.flatnonzero(~correct))

    # examples page: balanced on purpose so both kinds are visible (the summary line states the real accuracy)
    rng = random.Random(0)
    chosen = rng.sample(right, 24) + rng.sample(wrong, min(16, len(wrong)))
    rng.shuffle(chosen)
    acc = results["comparison"]["fine_tuned_calibrated"]["accuracy"]
    summary = (f"{len(chosen)} sampled test rows (balanced on purpose: the real test accuracy is {acc:.1%}, "
               f"{len(right)} right and {len(wrong)} wrong out of {len(test)}). Percentages are calibrated probabilities; a tick marks the true answer.")
    write_page("examples_template.html", {"examples": [make_example(int(i), test[i], probs[i]) for i in chosen], "summary": summary},
               os.path.join(ROOT, "examples.html"))

    # animation: a plain random sample, so its running score is an honest preview of the test accuracy
    sample = random.Random(1).sample(range(len(test)), N_ANIMATION)
    t = results["training"]
    write_page("animation_template.html", {
        "run": run_label(results),
        "train": [{"step": p["step"], "loss": p["loss"], "smooth": s} for p, s in zip(t["losses"], trailing_mean([p["loss"] for p in t["losses"]]))],
        "val": t["val_history"], "best_step": t["best_step"],
        "bars": [{**r, "label": "fine-tuned"} if "LoRA rank" in r["label"] else r          # the run's settings sit in the subtitle
                 for r in rows if "LoRA rank" not in r["label"] or r["kind"] == "main"],  # other runs left out
        "comparison": results["comparison"],
        "examples": [make_example(i, test[i], probs[i]) for i in sample],
    }, os.path.join(ROOT, "index.html"))
    print("wrote results/figures/*.png, examples.html, index.html")


if __name__ == "__main__":
    main()
