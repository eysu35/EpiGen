import copy

import torch
import torch.nn.functional as F


@torch.no_grad()
def _mean_loss(model, batches):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for tokens in batches:
        logits = model(tokens[:, :-1]).logits
        total_loss += F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="mean").item()
    model.train(was_training)
    return total_loss / len(batches)


def learner_step(model, optimizer, tokens):
    model.train()
    optimizer.zero_grad()
    logits = model(tokens[:, :-1]).logits
    loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="mean")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def compute_ppl_reward(pretrained_frozen, tokens):
    logits = pretrained_frozen(tokens[:, :-1]).logits
    return -F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="mean").item()


def compute_learner_reward(tokens, learner, S, lr_learner):
    return _loss_drop(learner, tokens, [tokens], S, lr_learner)


def compute_epiplexity_reward(
    tokens,
    learner,
    buf,
    S,
    lr_learner,
    local_sample_from_buffer,
    real_pool=None,
    real_sample_n=0,
    eval_real_only=False,
):
    sub_batch = tokens.shape[0]

    def _sample_real():
        idx = torch.randint(0, len(real_pool), (real_sample_n * sub_batch,))
        real_seqs = real_pool[idx].to(tokens.device)
        return [real_seqs[i : i + sub_batch] for i in range(0, real_sample_n * sub_batch, sub_batch)]

    if eval_real_only:
        assert real_pool is not None and real_sample_n > 0, "eval_real_only requires real_pool and real_sample_n > 0"
        eval_batches = _sample_real()
    else:
        eval_batches = buf.sample(local_sample_from_buffer, curr_batch=tokens)
        if real_pool is not None and real_sample_n > 0:
            eval_batches += _sample_real()

    return _loss_drop(learner, tokens, eval_batches, S, lr_learner)


def _loss_drop(learner, tokens, eval_batches, S, lr_learner):
    lrn = copy.deepcopy(learner)
    opt = torch.optim.AdamW(lrn.parameters(), lr=lr_learner, weight_decay=0.01)
    loss_before = _mean_loss(lrn, eval_batches)
    for _ in range(S):
        learner_step(lrn, opt, tokens)
    loss_after = _mean_loss(lrn, eval_batches)
    del lrn, opt
    return loss_before - loss_after
