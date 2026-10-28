"""Phase 1: turn SmolVLM-256M into a ScienceQA decision model (letter logits, LoRA, temperature). Load, train, test."""
import json
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolVLM-256M-Instruct")
OUT_DIR = os.environ.get("OUT_DIR", "/content/phase1_smol")
STAGE = os.environ.get("STAGE", "all")              # "all", or "finish" to calibrate + test the saved adapter
LETTERS = "ABCDE"
NEG = -1e9                  # logit for letters that are not valid options
LOAD_SIDE = 1024            # images are shrunk so their longest side is at most this when loaded
IMAGE_SIDE = os.environ.get("IMAGE_SIDE", "auto")   # 512 (one 64-token tile), 1024 (tiles + global view) or auto
LETTER_STYLE = os.environ.get("LETTER_STYLE", "auto")  # "space" (" A"), "bare" ("A") or auto
BATCH = 16
LR = float(os.environ.get("LR", 1e-4))
LORA_RANK = int(os.environ.get("LORA_RANK", 16))
WARMUP_STEPS = 20
EPOCHS = 4
EVAL_EVERY = 100            # steps between validation-subset checks
PATIENCE = 5                # stop after this many checks without a lower validation-subset loss
LOG_EVERY = 20
EVAL_SEEDS = (0, 1, 2)      # option-order shuffles for the test set; seed 0 is the main one
VAL_SUBSET = 500            # validation rows that pick the checkpoint; the rest fit the temperature
SEED = 0
NUM_WORKERS = min(6, os.cpu_count() or 1)
# bf16 only where the GPU has it natively (Ampere+); a T4 gets fp16 with a gradient scaler
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
T0 = time.time()


def log(msg):
    # Print a message with the elapsed minutes since the script started.
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def seed_everything(seed):
    # Seed python, numpy and torch so runs are repeatable.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def shrink(row):
    # Keep only the fields we need and shrink the image.
    image = row["image"].convert("RGB")
    image.thumbnail((LOAD_SIDE, LOAD_SIDE))
    keep = ("question", "choices", "answer", "hint", "subject", "grade")
    return {**{k: row[k] for k in keep}, "image": image}


def load_rows(split):
    # Load one ScienceQA split, keeping only rows that have an image.
    ds = load_dataset("derek-thomas/ScienceQA", split=split)
    return [shrink(r) for r in ds if r["image"] is not None]


def split_validation(val):
    # Fixed split: VAL_SUBSET rows pick the checkpoint, the remaining rows fit the temperature.
    order = list(range(len(val)))
    random.Random(SEED).shuffle(order)
    return [val[i] for i in order[:VAL_SUBSET]], [val[i] for i in order[VAL_SUBSET:]]


def blank_images(rows):
    # Copy of the rows with every image replaced by plain grey of the same size (does the model use the image?).
    return [{**r, "image": Image.new("RGB", r["image"].size, (128, 128, 128))} for r in rows]


def shuffle_choices(row, rng):
    # Return a copy of the row with choices shuffled and the answer index updated.
    order = list(range(len(row["choices"])))
    rng.shuffle(order)
    return {**row, "choices": [row["choices"][i] for i in order], "answer": order.index(row["answer"])}


def build_prompt(row):
    # Build the text prompt: optional context, question, lettered options, then "Answer:".
    options = "   ".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(row["choices"]))
    context = f"Context: {row['hint']}\n" if row["hint"] else ""
    return f"{context}Question: {row['question']}\n{options}\nAnswer:"


def render(processor, row):
    # Chat-template text for one row: image slot, prompt, then the "Assistant:" turn opener.
    message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_prompt(row)}]}]
    return processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


def collate(rows, processor, shuffle):
    # Turn rows into model inputs (image + chat-template text), labels and option counts.
    if shuffle:
        rows = [shuffle_choices(r, random) for r in rows]
    texts = [render(processor, r) for r in rows]
    batch = processor(text=texts, images=[[r["image"]] for r in rows], padding=True, return_tensors="pt")
    labels = torch.tensor([r["answer"] for r in rows])
    n_options = torch.tensor([len(r["choices"]) for r in rows])
    return batch, labels, n_options


def make_loader(rows, processor, shuffle, batch_size):
    # Build a seeded DataLoader whose workers do the image and text preprocessing.
    return DataLoader(rows, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=NUM_WORKERS,
                      generator=torch.Generator().manual_seed(SEED),
                      collate_fn=partial(collate, processor=processor, shuffle=shuffle))


def forever(loader):
    # Yield batches from the loader, starting a new epoch whenever it runs out.
    while True:
        yield from loader


def load_model():
    # Load the model in fp32 on the GPU; forward passes run under autocast in DTYPE.
    return AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda")


def load_processor(side):
    # Processor that resizes images to `side` on the longest edge (512 = no tiling), right padding.
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.size = {"longest_edge": side}
    processor.tokenizer.padding_side = "right"
    return processor


def get_letter_ids(tokenizer, style):
    # Token ids for A-E as the first answer token: " A" after "Assistant:" (space) or "A" (bare).
    prefix = " " if style == "space" else ""
    ids = []
    for c in LETTERS:
        enc = tokenizer.encode(prefix + c, add_special_tokens=False)
        assert len(enc) == 1, f"{prefix + c!r} is not a single token: {enc}"
        ids.append(enc[0])
    log(f"{style} letters: {dict(zip(LETTERS, ids))}")
    return ids


def letter_weights(model, processor, style):
    # Frozen lm_head rows for the letter tokens; these turn the last hidden state into option logits.
    return model.lm_head.weight[get_letter_ids(processor.tokenizer, style)].detach().float()


def to_cuda(batch):
    # Move a processor batch to the GPU (pixels stay fp32; autocast handles the cast).
    return {k: v.to("cuda") for k, v in batch.items()}


def option_logits(core, option_w, batch, n_options):
    # One forward pass -> logits for letters A-E at the last real token; invalid letters get NEG.
    batch = to_cuda(batch)
    with torch.autocast("cuda", dtype=DTYPE):
        hidden = core(**batch, use_cache=False).last_hidden_state
    positions = torch.arange(hidden.shape[1], device=hidden.device)
    last = (batch["attention_mask"] * positions).argmax(dim=1)
    logits = hidden[torch.arange(len(last)), last].float() @ option_w.T
    valid = torch.arange(len(LETTERS), device=logits.device)[None] < n_options.to(logits.device)[:, None]
    return logits.masked_fill(~valid, NEG)


def lr_at(step, total):
    # Linear warmup, then linear decay down to 10% of the base learning rate.
    warm = min(1.0, (step + 1) / WARMUP_STEPS)
    decay = 1 - 0.9 * max(0, step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return LR * warm * decay


def train_steps(core, option_w, batches, opt, scaler, params, start, end, total):
    # Run optimiser steps start..end-1; cross-entropy on the restricted softmax; returns logged losses.
    logged, window = [], []
    for step in range(start, end):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total)
        batch, labels, n_options = next(batches)
        loss = F.cross_entropy(option_logits(core, option_w, batch, n_options), labels.to("cuda"))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        window.append(loss.item())
        if (step + 1) % LOG_EVERY == 0:
            logged.append({"step": step + 1, "loss": float(np.mean(window))})
            log(f"step {step + 1}/{total} loss {logged[-1]['loss']:.4f} lr {lr_at(step, total):.2e}")
            window = []
    return logged


@torch.no_grad()
def evaluate(core, option_w, processor, rows, seed):
    # Score every row once with its options shuffled by `seed`; returns logits and metadata.
    shuffled = [shuffle_choices(r, random.Random(seed * 100003 + i)) for i, r in enumerate(rows)]
    all_logits = []
    for batch, _, n_options in make_loader(shuffled, processor, shuffle=False, batch_size=32):
        all_logits.append(option_logits(core, option_w, batch, n_options).cpu())
    return {
        "logits": torch.cat(all_logits).numpy(),
        "labels": np.array([r["answer"] for r in shuffled]),
        "n_options": np.array([len(r["choices"]) for r in shuffled]),
        "subjects": [r["subject"] for r in rows],
        "grades": [r["grade"] for r in rows],
    }


def accuracy(ev):
    # Fraction of rows whose highest logit is the true option.
    return float(np.mean(ev["logits"].argmax(1) == ev["labels"]))


def pick_setup(model, val_subset, results):
    # Zero-shot accuracy on the validation subset for each image side x letter style; keep the best pair.
    sides = (512, 1024) if IMAGE_SIDE == "auto" else (int(IMAGE_SIDE),)
    styles = ("space", "bare") if LETTER_STYLE == "auto" else (LETTER_STYLE,)
    sweep = []
    for side in sides:
        processor = load_processor(side)
        for style in styles:
            acc = accuracy(evaluate(model.model, letter_weights(model, processor, style), processor, val_subset, 0))
            sweep.append({"image_side": side, "letters": style, "accuracy": acc})
            log(f"zero-shot val subset, image side {side}, {style} letters: accuracy {acc:.4f}")
    best = max(sweep, key=lambda s: s["accuracy"])
    results["setup_sweep"] = sweep
    results["zero_shot_val_subset"] = {"n": len(val_subset), "accuracy": best["accuracy"]}
    results["config"].update(image_side=best["image_side"], letters=best["letters"])
    log(f"using image side {best['image_side']} with {best['letters']} letters")
    return best["image_side"], best["letters"]


def fit_temperature(ev):
    # Fit one scalar T (logits / T) that minimises negative log-likelihood on this set.
    logits, labels = torch.tensor(ev["logits"]), torch.tensor(ev["labels"])
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def val_check(peft_model, core, option_w, processor, val_subset):
    # Loss and accuracy of the current model on the validation subset (restores train mode afterwards).
    peft_model.eval()
    ev = evaluate(core, option_w, processor, val_subset, 0)
    peft_model.train()
    logits, labels = torch.tensor(ev["logits"]), torch.tensor(ev["labels"])
    return float(F.cross_entropy(logits, labels)), float((logits.argmax(1) == labels).float().mean())


def reliability(conf, correct, bins=10):
    # Per-bin confidence vs accuracy and the expected calibration error.
    edges = torch.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            gap = (correct[in_bin].mean() - conf[in_bin].mean()).abs()
            ece += float(in_bin.float().mean() * gap)
        table.append({"lo": float(lo), "hi": float(hi), "count": int(in_bin.sum()),
                      "confidence": float(conf[in_bin].mean()) if in_bin.any() else None,
                      "accuracy": float(correct[in_bin].mean()) if in_bin.any() else None})
    return table, ece


def accuracy_by(correct, keys):
    # Accuracy and count for each distinct value in keys.
    keys = np.array(keys)
    return {str(k): {"n": int((keys == k).sum()), "accuracy": float(correct[keys == k].mean())} for k in sorted(set(keys))}


def metrics(ev, T):
    # All test metrics for one evaluation at temperature T.
    probs = torch.softmax(torch.tensor(ev["logits"]) / T, dim=1)
    labels = torch.tensor(ev["labels"])
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    table, ece = reliability(conf, correct)
    nll = -torch.log(probs[torch.arange(len(labels)), labels].clamp_min(1e-12)).mean()
    acc = float(correct.mean())
    return {
        "n": len(labels),
        "accuracy": acc,
        "accuracy_ci95": 1.96 * float(np.sqrt(acc * (1 - acc) / len(labels))),   # +- half-width, normal approximation
        "chance_accuracy": float(np.mean(1 / ev["n_options"])),                  # expected accuracy of a random pick
        "by_subject": accuracy_by(correct.numpy(), ev["subjects"]),
        "by_grade": accuracy_by(correct.numpy(), ev["grades"]),
        "by_n_options": accuracy_by(correct.numpy(), ev["n_options"]),
        "ece": ece,
        "nll": float(nll),
        "mean_confidence_right": float(conf[correct == 1].mean()),
        "mean_confidence_wrong": float(conf[correct == 0].mean()),
        "reliability": table,
        "predicted_letter_share": (torch.bincount(pred, minlength=len(LETTERS)).float() / len(pred)).tolist(),
        "true_letter_share": (torch.bincount(labels, minlength=len(LETTERS)).float() / len(labels)).tolist(),
    }


def report(evals, T):
    # Full metrics for the main order (seed 0) plus accuracy under every reshuffled order.
    main_seed = EVAL_SEEDS[0]
    acc = {str(s): metrics(evals[s], T)["accuracy"] for s in EVAL_SEEDS}
    return {"T": T, "main": metrics(evals[main_seed], T), "accuracy_by_order_seed": acc,
            "max_accuracy_change_vs_seed0": max(abs(a - acc[str(main_seed)]) for a in acc.values())}


def test_all_orders(core, option_w, processor, test):
    # Evaluate the test set under every option-order seed.
    evals = {}
    for seed in EVAL_SEEDS:
        evals[seed] = evaluate(core, option_w, processor, test, seed)
        log(f"test order seed {seed}: accuracy {accuracy(evals[seed]):.4f}")
    return evals


def image_ablation(core, option_w, processor, test):
    # Test accuracy (seed-0 order) when every image is blanked: the gap to the real accuracy is what the image adds.
    acc = accuracy(evaluate(core, option_w, processor, blank_images(test), EVAL_SEEDS[0]))
    log(f"test with blank images: accuracy {acc:.4f}")
    return acc


def save_json(results, name="results.json"):
    # Write the results dict to the output directory.
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(results, f, indent=2)


def comparison_row(rep):
    # Pick the headline numbers for the comparison table.
    m = rep["main"]
    return {"accuracy": m["accuracy"], "accuracy_ci95": m["accuracy_ci95"], "chance_accuracy": m["chance_accuracy"],
            "accuracy_without_image": rep["accuracy_without_image"], "ece": m["ece"], "nll": m["nll"],
            "mean_confidence_right": m["mean_confidence_right"], "mean_confidence_wrong": m["mean_confidence_wrong"],
            "accuracy_by_order_seed": rep["accuracy_by_order_seed"], "T": rep["T"]}


def calibrate_and_test(peft_model, core, option_w, processor, val_calib, test, results, timings):
    # Fit T on the calibration part of validation, then test before/after calibration and under reshuffled option orders.
    peft_model.eval()
    calib_ev = evaluate(core, option_w, processor, val_calib, 0)
    T = fit_temperature(calib_ev)
    results["temperature"] = T
    results["validation_calibration"] = {"n": len(val_calib), "accuracy": accuracy(calib_ev)}
    log(f"fitted T = {T:.4f} on {len(val_calib)} calibration rows")
    timings["calibration_done"] = (time.time() - T0) / 60

    ft_test = test_all_orders(core, option_w, processor, test)
    without_image = image_ablation(core, option_w, processor, test)
    results["fine_tuned"] = {**report(ft_test, 1.0), "accuracy_without_image": without_image}
    results["fine_tuned_calibrated"] = {**report(ft_test, T), "accuracy_without_image": without_image}
    results["comparison"] = {"zero_shot": comparison_row(results["zero_shot"]),
                             "fine_tuned": comparison_row(results["fine_tuned"]),
                             "fine_tuned_calibrated": comparison_row(results["fine_tuned_calibrated"])}
    timings["test_done"] = (time.time() - T0) / 60
    np.savez(os.path.join(OUT_DIR, "logits.npz"), val=calib_ev["logits"], val_labels=calib_ev["labels"],
             test=ft_test[EVAL_SEEDS[0]]["logits"], test_labels=ft_test[EVAL_SEEDS[0]]["labels"])
    save_json(results)
    log("done")
    print(json.dumps(results["comparison"], indent=2))


def finish_from_saved_adapter():
    # Calibrate and test the adapter saved in OUT_DIR/adapter (used when training was stopped by hand).
    with open(os.path.join(OUT_DIR, "results.json")) as f:
        results = json.load(f)
    results["training"]["stopped_by_hand"] = True
    seed_everything(SEED)
    _, val_calib = split_validation(load_rows("validation"))
    test = load_rows("test")
    model = load_model()
    processor = load_processor(results["config"]["image_side"])
    option_w = letter_weights(model, processor, results["config"]["letters"])
    core = model.model
    peft_model = PeftModel.from_pretrained(model, os.path.join(OUT_DIR, "adapter"))
    calibrate_and_test(peft_model, core, option_w, processor, val_calib, test, results,
                       results.setdefault("finish_timings_min", {}))


def main():
    # Setup sweep -> zero-shot baseline -> LoRA training -> temperature fit -> test, saving results as we go.
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_everything(SEED)
    results = {"model": MODEL_ID, "config": {"batch": BATCH, "lr": LR, "lora_rank": LORA_RANK, "epochs": EPOCHS,
                                             "seed": SEED, "dtype": str(DTYPE)}, "timings_min": {}}

    train, val, test = (load_rows(s) for s in ("train", "validation", "test"))
    results["sizes_with_image"] = {"train": len(train), "validation": len(val), "test": len(test)}
    log(f"rows with images: {results['sizes_with_image']}")
    val_subset, val_calib = split_validation(val)
    model = load_model()
    model.eval()
    side, style = pick_setup(model, val_subset, results)
    processor = load_processor(side)
    option_w = letter_weights(model, processor, style)
    core = model.model
    log(f"example prompt: {render(processor, test[0])!r}")
    results["timings_min"]["setup"] = (time.time() - T0) / 60

    # 1. zero-shot baseline on the full test set
    zs_test = test_all_orders(core, option_w, processor, test)
    results["zero_shot"] = {**report(zs_test, 1.0), "accuracy_without_image": image_ablation(core, option_w, processor, test)}
    results["timings_min"]["zero_shot"] = (time.time() - T0) / 60
    save_json(results)

    # 2. LoRA training for EPOCHS epochs, checking the validation subset every EVAL_EVERY steps
    peft_model = get_peft_model(model, LoraConfig(
        r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.05,
        target_modules=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
    peft_model.print_trainable_parameters()
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=DTYPE == torch.float16)
    batches = forever(make_loader(train, processor, shuffle=True, batch_size=BATCH))
    total = EPOCHS * (len(train) // BATCH)
    results["training"] = {"planned_steps": total}
    log(f"training up to {total} steps ({EPOCHS} epochs)")
    peft_model.train()
    train_start = time.time()
    best, history, losses, stale = {"loss": float("inf")}, [], [], 0
    for start in range(0, total, EVAL_EVERY):
        end = min(start + EVAL_EVERY, total)
        losses += train_steps(core, option_w, batches, opt, scaler, params, start, end, total)
        val_loss, val_acc = val_check(peft_model, core, option_w, processor, val_subset)
        history.append({"step": end, "val_loss": val_loss, "val_accuracy": val_acc})
        log(f"step {end}: validation-subset loss {val_loss:.4f} accuracy {val_acc:.4f}")
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "step": end, "state": {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(peft_model).items()}}
            peft_model.save_pretrained(os.path.join(OUT_DIR, "adapter"))
            stale = 0
        else:
            stale += 1
        results["training"].update({"steps_done": end, "seconds_per_step": (time.time() - train_start) / end,
                                    "losses": losses, "val_history": history, "best_step": best["step"]})
        save_json(results)
        if stale >= PATIENCE:
            log(f"no lower validation-subset loss for {PATIENCE} checks, stopping early")
            break
    set_peft_model_state_dict(peft_model, best["state"])
    log(f"using the adapter from step {best['step']} (lowest validation-subset loss)")
    results["timings_min"]["training_done"] = (time.time() - T0) / 60

    calibrate_and_test(peft_model, core, option_w, processor, val_calib, test, results, results["timings_min"])


if __name__ == "__main__":
    finish_from_saved_adapter() if STAGE == "finish" else main()
