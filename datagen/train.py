import copy
import os
import random
import time
from collections import Counter

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from .buffer import Buffer
from .reward import compute_epiplexity_reward, compute_learner_reward, compute_ppl_reward, learner_step


@torch.no_grad()
def generate_batch(model, batch_size, seq_len, temperature, device, prompt_ids):
    model.eval()
    prompt_len = len(prompt_ids)
    prompt_batch = prompt_ids.unsqueeze(0).expand(batch_size, -1).clone()

    past_key_values = None
    if prompt_len > 1:
        outputs = model(prompt_batch[:, :-1], use_cache=True)
        past_key_values = outputs.past_key_values
    input_ids = prompt_batch[:, -1:]

    generated = []
    for _ in range(seq_len):
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :] / temperature
        next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)
        generated.append(next_token)
        input_ids = next_token

    model.train()
    return torch.cat(generated, dim=1)


def generator_reinforce_step(model, optimizer, tokens, advantage):
    optimizer.zero_grad()
    model.train()
    logits = model(tokens[:, :-1]).logits
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1).mean()
    (-advantage * token_log_probs).backward()
    params = model.module.parameters() if hasattr(model, "module") else model.parameters()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()


@torch.no_grad()
def compute_generator_entropy(model, tokens):
    was_training = model.training
    model.eval()
    logits = model(tokens[:, :-1]).logits
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
    model.train(was_training)
    return entropy.item()


def compute_text_stats(tokens):
    flat = tokens.reshape(-1).tolist()
    unique = len(set(flat))
    bigrams = [(flat[i], flat[i + 1]) for i in range(len(flat) - 1)]
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
    bigram_repeat_rate = repeated_bigrams / len(bigram_counts)
    return {
        "unique_tokens": unique,
        "unique_token_ratio": unique / len(flat),
        "bigram_repeat_rate": bigram_repeat_rate,
    }


@torch.no_grad()
def compute_pretrained_ppl(pretrained_model, tokens):
    logits = pretrained_model(tokens[:, :-1]).logits
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1), reduction="mean")
    return loss.item()


def save_samples(tokenizer, tokens, global_step, out_dir, n_samples=4):
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    path = os.path.join(samples_dir, f"step_{global_step:06d}.txt")
    with open(path, "w") as f:
        f.write(f"=== Step {global_step} ===\n\n")
        for i in range(min(n_samples, tokens.shape[0])):
            text = tokenizer.decode(tokens[i].tolist(), skip_special_tokens=False)
            f.write(f"--- Sample {i + 1} ---\n{text}\n\n")


def load_real_train_pool(tokenizer, n_tokens, rank, world_size, seq_len):
    cache_dir = os.environ.get("OWT_CACHE_DIR", "cache")
    cache_path = f"{cache_dir}/owt_train_tokens_{n_tokens // 1_000_000}M.npy"
    if os.path.exists(cache_path):
        tokens = np.load(cache_path)[:n_tokens]
    else:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
        all_tokens = []
        for example in ds:
            all_tokens.extend(tokenizer.encode(example["text"]))
            if len(all_tokens) >= n_tokens:
                break
        tokens = np.array(all_tokens[:n_tokens], dtype=np.int32)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp_path = cache_path[:-4] + f".tmp{os.getpid()}.npy"
        np.save(tmp_path, tokens)
        os.chmod(tmp_path, 0o644)
        os.rename(tmp_path, cache_path)
    n_seqs = len(tokens) // seq_len
    seqs = torch.from_numpy(tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len).astype(np.int64))
    per_rank = n_seqs // world_size
    assert per_rank > 0, "real_train_tokens is too small to fill one sequence per rank"
    return seqs[rank * per_rank : (rank + 1) * per_rank]


def real_train_iter(pool, sub_batch):
    i = 0
    while True:
        yield pool[[(i + j) % len(pool) for j in range(sub_batch)]]
        i = (i + sub_batch) % len(pool)


def mixed_batch(tokens, real_iter, n_real, device):
    """Learner batch: n_real real sequences plus the rest of the generated batch."""
    if n_real == 0:
        return tokens
    real = next(real_iter).to(device)
    if n_real >= tokens.shape[0]:
        return real
    return torch.cat([real, tokens[: tokens.shape[0] - n_real]])


def load_eval_data(tokenizer, max_tokens, offset=0):
    cache_dir = os.environ.get("OWT_CACHE_DIR", "cache")
    suffix = f"_off{offset // 1_000_000}M" if offset else ""
    cache_path = f"{cache_dir}/owt_tokens_{max_tokens // 1_000_000}M{suffix}.npy"
    if os.path.exists(cache_path):
        tokens = np.load(cache_path)[:max_tokens]
    else:
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
        all_tokens = []
        for example in ds:
            all_tokens.extend(tokenizer.encode(example["text"]))
            if len(all_tokens) >= offset + max_tokens:
                break
        tokens = np.array(all_tokens[offset : offset + max_tokens], dtype=np.int32)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp_path = cache_path[:-4] + f".tmp{os.getpid()}.npy"
        np.save(tmp_path, tokens)
        os.chmod(tmp_path, 0o644)
        os.rename(tmp_path, cache_path)
    seq_len = 512 + 1
    n_seqs = len(tokens) // seq_len
    tokens = tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)
    return torch.from_numpy(tokens.astype(np.int64))


@torch.no_grad()
def eval_model(model, eval_data, device, batch_size=64):
    model.eval()
    losses = []
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i : i + batch_size].to(device)
        logits = model(batch[:, :-1]).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
            reduction="mean",
        )
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


@hydra.main(config_path="../configs/glue_table", config_name="epigen", version_base=None)
def main(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    if world_size > 1:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    assert cfg.batch_size % world_size == 0, "batch_size must be divisible by world_size"
    local_batch_size = cfg.batch_size // world_size
    local_sample_from_buffer = max(1, cfg.sample_from_buffer // world_size)

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if is_main and not cfg.no_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.tag if cfg.tag else None,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.tag] if cfg.tag else None,
        )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    pretrained_weights = GPT2LMHeadModel.from_pretrained("gpt2")

    prompt_str = cfg.get("prompt", "")
    if prompt_str:
        prompt_ids = torch.tensor(tokenizer.encode(prompt_str), dtype=torch.long, device=device)
    else:
        prompt_ids = torch.tensor([tokenizer.eos_token_id], dtype=torch.long, device=device)

    pretrained_frozen = copy.deepcopy(pretrained_weights).to(device)
    pretrained_frozen.eval()
    for p in pretrained_frozen.parameters():
        p.requires_grad = False

    init_weights = GPT2LMHeadModel(GPT2Config()) if cfg.get("random_init", False) else pretrained_weights

    if cfg.get("generator_init_path", ""):
        generator = GPT2LMHeadModel.from_pretrained(cfg.generator_init_path).to(device)
    else:
        generator = copy.deepcopy(init_weights).to(device)
    generator.gradient_checkpointing_enable()
    if world_size > 1:
        generator_ddp = DDP(generator, device_ids=[local_rank] if torch.cuda.is_available() else None)
    else:
        generator_ddp = generator
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=cfg.lr_generator, weight_decay=0.01)

    real_frac = float(cfg.get("real_frac", 0.0))
    assert 0.0 <= real_frac <= 1.0, "real_frac must be in [0, 1]"
    n_real = round(real_frac * local_batch_size)
    assert real_frac == 0.0 or n_real > 0, "real_frac is too small to contribute a sequence to the learner batch"

    eval_offset = cfg.get("eval_offset", 0)
    real_train_tokens = cfg.get("real_train_tokens", 100_000_000)
    if n_real > 0:
        assert eval_offset >= real_train_tokens, (
            "eval_offset must be >= real_train_tokens, else the held-out eval overlaps the OWT train pool"
        )
    eval_data = load_eval_data(tokenizer, cfg.eval_tokens, eval_offset)
    baseline = 0.0
    global_step = 0
    last_eval_step = 0
    last_ckpt_step = 0

    real_iter = None
    if n_real > 0:
        train_pool = load_real_train_pool(tokenizer, real_train_tokens, rank, world_size, cfg.seq_len)
        real_iter = real_train_iter(train_pool, n_real)

    if cfg.get("learner_init_path", ""):
        learner = GPT2LMHeadModel.from_pretrained(cfg.learner_init_path).to(device)
    else:
        learner = copy.deepcopy(init_weights).to(device)
    learner.gradient_checkpointing_enable()
    learner_optimizer = torch.optim.AdamW(learner.parameters(), lr=cfg.lr_learner, weight_decay=0.01)
    buf = Buffer()

    epi_ceiling, epi_total_n = 0.0, 0

    if is_main:
        step0_learner_loss = eval_model(learner, eval_data, device)
        step0_generator_loss = eval_model(generator, eval_data, device)
        print(f"Step 0 eval loss: learner={step0_learner_loss:.4f}, generator={step0_generator_loss:.4f}")
        if not cfg.no_wandb:
            wandb.log({"eval/learner_loss": step0_learner_loss, "eval/generator_loss": step0_generator_loss}, step=0)

    for t in range(1, cfg.H + 1):
        step_t0 = time.time()

        generator.gradient_checkpointing_disable()
        gen_t0 = time.time()
        tokens = generate_batch(generator, local_batch_size, cfg.seq_len, cfg.tau, device, prompt_ids)
        gen_time = time.time() - gen_t0
        generator.gradient_checkpointing_enable()

        if cfg.reward_mode == "pretrained_ppl":
            reward = compute_ppl_reward(pretrained_frozen, tokens)
        elif cfg.reward_mode == "no_buffer":
            reward = compute_learner_reward(tokens, learner, cfg.S, cfg.lr_learner)
        else:
            needs_real_pool = cfg.get("real_sample_n", 0) > 0 or cfg.get("eval_real_only", False)
            reward = compute_epiplexity_reward(
                tokens,
                learner,
                buf,
                cfg.S,
                cfg.lr_learner,
                local_sample_from_buffer,
                real_pool=eval_data if needs_real_pool else None,
                real_sample_n=cfg.get("real_sample_n", 0),
                eval_real_only=cfg.get("eval_real_only", False),
            )

        learner_losses = [
            learner_step(learner, learner_optimizer, mixed_batch(tokens, real_iter, n_real, device))
            for _ in range(cfg.S)
        ]

        advantage = reward - baseline

        if not cfg.freeze_generator:
            generator_reinforce_step(generator_ddp, generator_optimizer, tokens, advantage)

        curr_loss = float(learner_losses[-1])
        epi_ceiling += local_batch_size * curr_loss
        epi_total_n += local_batch_size
        learner_epi = epi_ceiling - curr_loss * epi_total_n

        r_sync = reward
        if world_size > 1:
            r_t = torch.tensor(reward, device=device)
            dist.all_reduce(r_t, op=dist.ReduceOp.SUM)
            r_sync = r_t.item() / world_size
        baseline = cfg.beta * baseline + (1 - cfg.beta) * r_sync

        if cfg.reward_mode == "epiplexity":
            buf.add(tokens)
        global_step += cfg.S
        step_time = time.time() - step_t0

        if is_main and (t % cfg.log_interval == 0 or t == 1):
            entropy = compute_generator_entropy(generator, tokens)
            text_stats = compute_text_stats(tokens.cpu())
            pretrained_ppl = compute_pretrained_ppl(pretrained_frozen, tokens)

            if not cfg.no_wandb:
                wandb.log(
                    {
                        "reward": reward,
                        "advantage": advantage,
                        "baseline": baseline,
                        "learner_train_loss": curr_loss,
                        "learner_epi": learner_epi,
                        "generator_entropy": entropy,
                        "generator_pretrained_ppl": pretrained_ppl,
                        "text/unique_token_ratio": text_stats["unique_token_ratio"],
                        "text/bigram_repeat_rate": text_stats["bigram_repeat_rate"],
                        "n_buffer": len(buf),
                        "perf/step_time": step_time,
                        "perf/gen_time": gen_time,
                    },
                    step=global_step,
                )

        if global_step - last_eval_step >= cfg.eval_interval:
            last_eval_step = global_step
            if world_size > 1:
                dist.barrier()
            learner_eval = eval_model(learner, eval_data, device)
            generator_eval = eval_model(generator, eval_data, device)
            if is_main:
                save_samples(tokenizer, tokens.cpu(), global_step, cfg.out_dir)
            if is_main and not cfg.no_wandb:
                wandb.log(
                    {
                        "eval/learner_loss": learner_eval,
                        "eval/generator_loss": generator_eval,
                        "n_buffer": len(buf),
                    },
                    step=global_step,
                )

        ckpt_interval = cfg.get("checkpoint_interval", 0)
        if is_main and ckpt_interval > 0 and global_step - last_ckpt_step >= ckpt_interval:
            last_ckpt_step = global_step
            learner.save_pretrained(os.path.join(cfg.out_dir, f"learner_step{global_step:06d}"))

    if is_main:
        final_learner = eval_model(learner, eval_data, device)
        final_generator = eval_model(generator, eval_data, device)
        print(f"Final loss: learner={final_learner:.4f}, generator={final_generator:.4f}")

        learner.save_pretrained(os.path.join(cfg.out_dir, "learner"))
        generator.save_pretrained(os.path.join(cfg.out_dir, "generator"))
        print(f"Models saved to {cfg.out_dir}")

        if not cfg.no_wandb:
            wandb.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
