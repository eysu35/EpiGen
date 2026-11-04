import json
import os
import random

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from omegaconf import DictConfig
from transformers import GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer

_GLUE_TASKS = {
    "sst2": {"text_fields": ["sentence"], "num_labels": 2, "metric": "accuracy", "val_split": "validation"},
    "rte": {
        "text_fields": ["sentence1", "sentence2"],
        "num_labels": 2,
        "metric": "accuracy",
        "val_split": "validation",
    },
    "cola": {"text_fields": ["sentence"], "num_labels": 2, "metric": "matthews", "val_split": "validation"},
    "mrpc": {
        "text_fields": ["sentence1", "sentence2"],
        "num_labels": 2,
        "metric": "accuracy",
        "val_split": "validation",
    },
    "qnli": {"text_fields": ["question", "sentence"], "num_labels": 2, "metric": "accuracy", "val_split": "validation"},
    "qqp": {
        "text_fields": ["question1", "question2"],
        "num_labels": 2,
        "metric": "accuracy",
        "val_split": "validation",
    },
    "mnli": {
        "text_fields": ["premise", "hypothesis"],
        "num_labels": 3,
        "metric": "accuracy",
        "val_split": "validation_matched",
    },
}


def _mcc(y_true, y_pred):
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def _glue_tokenize(tok, examples, text_fields, max_len):
    if len(text_fields) == 1:
        return tok(examples[text_fields[0]], truncation=True, max_length=max_len, padding="max_length")
    return tok(
        examples[text_fields[0]], examples[text_fields[1]], truncation=True, max_length=max_len, padding="max_length"
    )


def get_tokens(hf_path, subset, split, field, cache_path, max_tokens, tok, offset=0):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    ds = load_dataset(
        hf_path, **({} if subset is None else {"name": subset}), split=split, streaming=True, trust_remote_code=True
    )
    tokens = []
    for ex in ds:
        tokens.extend(tok.encode(ex[field]))
        if len(tokens) >= offset + max_tokens:
            break
    arr = np.array(tokens[offset : offset + max_tokens], dtype=np.int32)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_path = cache_path[:-4] + f".tmp{os.getpid()}.npy"
    np.save(tmp_path, arr)
    os.chmod(tmp_path, 0o644)
    try:
        os.rename(tmp_path, cache_path)
    except OSError:
        os.remove(tmp_path)
    return arr


@torch.no_grad()
def eval_tokens(model, tokens, seq_len, batch_size, device):
    sl = seq_len + 1
    seqs = torch.from_numpy(tokens[: (len(tokens) // sl) * sl].reshape(-1, sl).astype(np.int64))
    losses = [
        F.cross_entropy(
            model(seqs[i : i + batch_size, :-1].to(device)).logits.flatten(0, 1),
            seqs[i : i + batch_size, 1:].to(device).flatten(),
        ).item()
        for i in range(0, len(seqs), batch_size)
    ]
    loss = float(np.mean(losses))
    return {"loss": round(loss, 4), "ppl": round(float(np.exp(loss)), 4)}


def eval_glue(path, tasks, n_epochs, lr, batch_size, max_seq_len, device, seed=42):
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    results = {}
    for task_name in tasks:
        task = _GLUE_TASKS[task_name]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = GPT2ForSequenceClassification.from_pretrained(path, num_labels=task["num_labels"])
        model.config.pad_token_id = tok.pad_token_id
        model = model.to(device)

        def tok_fn(ex, text_fields=task["text_fields"]):
            return _glue_tokenize(tok, ex, text_fields, max_seq_len)

        cols = ["input_ids", "attention_mask", "label"]

        # keep_in_memory: concurrent eval jobs otherwise race on the shared on-disk map cache (SIGBUS on GPFS).
        ds = load_dataset("nyu-mll/glue", task_name)
        train_ds = ds["train"].map(tok_fn, batched=True, keep_in_memory=True).with_format("torch", columns=cols)
        val_split = task.get("val_split", "validation")
        val_ds = ds[val_split].map(tok_fn, batched=True, keep_in_memory=True).with_format("torch", columns=cols)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)
        )
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

        total_steps = len(train_loader) * n_epochs
        warmup_steps = total_steps // 10
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s, warmup=warmup_steps, total=total_steps: (
                s / max(1, warmup) if s < warmup else max(0.0, (total - s) / max(1, total - warmup))
            ),
        )

        for _ in range(n_epochs):
            model.train()
            for batch in train_loader:
                loss = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["label"].to(device),
                ).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(batch["label"].tolist())

        score = (
            _mcc(all_labels, all_preds)
            if task["metric"] == "matthews"
            else float(np.mean(np.array(all_preds) == np.array(all_labels)))
        )
        results[task_name] = {"score": round(score, 4), "metric": task["metric"]}

        del model
        torch.cuda.empty_cache()

    return results


def eval_harness(path, tasks, batch_size, limit, device):
    lm = HFLM(pretrained=path, tokenizer="gpt2", batch_size=batch_size, device=device)
    res = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        task_manager=TaskManager(),
        limit=limit,
        log_samples=False,
    )
    return {
        t: {
            "acc": (r := res["results"].get(t, {})).get(
                "acc,none",
                r.get(
                    "acc_norm,none",
                    r.get("exact_match,none", r.get("exact_match,flexible-extract", r.get("exact_match,strict-match"))),
                ),
            ),
            "stderr": r.get(
                "acc_stderr,none",
                r.get(
                    "acc_norm_stderr,none",
                    r.get(
                        "exact_match_stderr,none",
                        r.get("exact_match_stderr,flexible-extract", r.get("exact_match_stderr,strict-match")),
                    ),
                ),
            ),
        }
        for t in tasks
    }


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    device = cfg.device
    evals = set(cfg.evals)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    n = cfg.get("max_tokens", 5_000_000)

    datasets = {}
    if "openwebtext" in evals:
        owt_off = cfg.get("owt_offset", 0)
        owt_suffix = f"_off{owt_off // 1_000_000}M" if owt_off else ""
        datasets["openwebtext"] = get_tokens(
            "Skylion007/openwebtext",
            None,
            "train",
            "text",
            f"{cfg.cache_dir}/owt_{n // 1_000_000}M{owt_suffix}.npy",
            n,
            tok,
            owt_off,
        )
    if "c4" in evals:
        datasets["c4"] = get_tokens(
            "allenai/c4", "en", "validation", "text", f"{cfg.cache_dir}/c4_{n // 1_000_000}M.npy", n, tok
        )
    if "slimpajama" in evals:
        datasets["slimpajama"] = get_tokens(
            "DKYoon/SlimPajama-6B",
            None,
            "validation",
            "text",
            f"{cfg.cache_dir}/slimpajama_{n // 1_000_000}M.npy",
            n,
            tok,
        )
    if "fineweb" in evals:
        datasets["fineweb"] = get_tokens(
            "HuggingFaceFW/fineweb",
            "sample-10BT",
            "train",
            "text",
            f"{cfg.cache_dir}/fineweb_{n // 1_000_000}M.npy",
            n,
            tok,
        )

    results = {"runs": {}}
    if os.path.exists(cfg.output):
        with open(cfg.output) as f:
            results["runs"].update(json.load(f).get("runs", {}))

    only = cfg.get("only", None)
    for r in cfg.runs:
        if only is not None and r.label != only:
            continue
        key = r.label
        if key in results["runs"] and not cfg.get("force", False):
            continue
        if r.path != "gpt2" and not os.path.isdir(r.path):
            print(f"missing: {r.path}")
            continue

        rd = {}
        for name, tokens in datasets.items():
            model = GPT2LMHeadModel.from_pretrained(r.path).eval().to(device)
            rd[name] = eval_tokens(model, tokens, cfg.seq_len, cfg.batch_size, device)
            del model
            torch.cuda.empty_cache()

        if "harness" in evals:
            rd["harness"] = eval_harness(
                r.path, list(cfg.harness_tasks), cfg.harness_batch_size, cfg.get("harness_limit"), device
            )

        if "glue" in evals:
            rd["glue"] = eval_glue(
                r.path,
                list(cfg.glue_tasks),
                cfg.get("glue_n_epochs", 3),
                cfg.get("glue_lr", 2e-5),
                cfg.get("glue_batch_size", 32),
                cfg.get("glue_max_seq_len", 128),
                device,
                cfg.get("seed", 42),
            )

        results["runs"][key] = rd
        os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
        with open(cfg.output, "w") as f:
            json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
