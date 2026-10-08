#!/usr/bin/env python3
"""
LLM Prompt Tuning Server (SGFPT) for SecP-Tuning comparison.

Implements BBT-style Forward-only Tuning (FoT) for RoBERTa-LARGE using
sep-CMA-ES in intrinsic subspace.  Supports:
  --plaintext : local CMA-ES (no MPC)
  default     : TCP server receiving prompts from C++ MPC client

Datasets: SST-2, Yelp Polarity, AG's News, MRPC, RTE

Multi-task usage (model loaded once):
  python llm_server.py --tasks sst2,yelp_polarity,ag_news,mrpc,rte
"""

import argparse
import os
import sys
import signal
import time as _time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import threading
from protocol import PromptServer, PromptConnection, fixed_to_float


# ─────────────────────────────────────────────────────────────────
# MPC helpers
# ─────────────────────────────────────────────────────────────────

def _recv_share_thread(conn: PromptConnection, result: list, idx: int) -> None:
    try:
        result[idx] = conn.recv_raw_share()
    except Exception as e:
        result[idx] = e


def recv_two_shares(conn0: PromptConnection, conn1: PromptConnection):
    """Receive duplicate zero-mask simulation values and use the first copy."""
    results = [None, None]
    t0 = threading.Thread(target=_recv_share_thread, args=(conn0, results, 0))
    t1 = threading.Thread(target=_recv_share_thread, args=(conn1, results, 1))
    t0.start(); t1.start()
    t0.join();  t1.join()
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            raise RuntimeError(f"Party {i} share recv failed: {r}") from r
    raw0, lambda_, d, bw, scale, party_num = results[0]
    raw1, lambda1, d1, bw1, scale1, pnum1 = results[1]
    if (lambda1, d1, bw1, scale1, pnum1) != (lambda_, d, bw, scale, party_num):
        raise ValueError("Party header mismatch")
    if not np.array_equal(raw0, raw1):
        raise ValueError("Simulation prompt copies differ between the two parties")
    # Simulation mode: raw0 IS the plaintext (masks=0)
    prompts = fixed_to_float(raw0, bw, scale).reshape(lambda_, d)
    return prompts, lambda_, d, bw, scale, party_num


def expand_fitness(fitnesses: np.ndarray, party_num: int) -> np.ndarray:
    """Split each fitness into party_num equal parts (aggregateRowsKernel sums them)."""
    return np.repeat(fitnesses / party_num, party_num)


# ─────────────────────────────────────────────────────────────────
# BBT-style Prompt Model
# ─────────────────────────────────────────────────────────────────

class PromptRoBERTa(torch.nn.Module):
    """RoBERTa-LARGE with continuous prompt embedding optimized via CMA-ES.

    Architecture (BBT):
      z in R^d  -->  A * z  -->  reshape to (L, emb_dim)  -->  prompt tokens
      prompt tokens prepended to input  -->  RoBERTa forward  -->  <mask> logits
    """

    def __init__(self, model_name: str, n_prompt_tokens: int, intrinsic_dim: int,
                 device: str, seed: int = 42):
        super().__init__()
        from transformers import RobertaForMaskedLM, RobertaTokenizer

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad = False

        self.device = device
        self.n_prompt_tokens = n_prompt_tokens
        self.intrinsic_dim = intrinsic_dim
        self.emb_dim = self.model.config.hidden_size  # 1024 for roberta-large
        self.prompt_dim = n_prompt_tokens * self.emb_dim

        rng = np.random.default_rng(seed)
        self.A = torch.tensor(
            rng.uniform(-1, 1, (self.prompt_dim, intrinsic_dim)),
            dtype=torch.float32, device=device
        )
        self.p0 = torch.zeros(self.prompt_dim, dtype=torch.float32, device=device)
        self.mask_token_id = self.tokenizer.mask_token_id

    @torch.no_grad()
    def set_prompt(self, z: np.ndarray):
        z_t = torch.tensor(z, dtype=torch.float32, device=self.device)
        self.prompt_emb = (self.p0 + self.A @ z_t).view(self.n_prompt_tokens, self.emb_dim)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                mask_positions: torch.Tensor):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.model.roberta.embeddings.word_embeddings(input_ids)
        prompt = self.prompt_emb.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prompt, inputs_embeds], dim=1)
        prompt_mask = torch.ones(
            batch_size, self.n_prompt_tokens,
            dtype=attention_mask.dtype, device=self.device
        )
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        shifted_mask_pos = mask_positions + self.n_prompt_tokens
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        return logits[torch.arange(batch_size, device=self.device), shifted_mask_pos]


# ─────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────

def load_dataset_splits(task_name: str, cache_dir: Optional[str] = None):
    from datasets import load_dataset
    cache_dir = cache_dir or os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
    os.makedirs(cache_dir, exist_ok=True)

    if task_name == "sst2":
        ds = load_dataset("glue", "sst2", cache_dir=cache_dir)
        return ds["train"], ds["validation"]
    elif task_name == "yelp_polarity":
        ds = load_dataset("yelp_polarity", cache_dir=cache_dir)
        return ds["train"], ds["test"]
    elif task_name == "ag_news":
        ds = load_dataset("ag_news", cache_dir=cache_dir)
        return ds["train"], ds["test"]
    elif task_name == "mrpc":
        ds = load_dataset("glue", "mrpc", cache_dir=cache_dir)
        return ds["train"], ds["validation"]
    elif task_name == "rte":
        ds = load_dataset("super_glue", "rte", cache_dir=cache_dir)
        return ds["train"], ds["validation"]
    else:
        raise ValueError(f"Unknown task: {task_name}")


class PromptTuningDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, task_name: str, tokenizer, max_seq_length: int = 128):
        self.data = hf_dataset
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.task_name == "sst2":
            text = f"{item['sentence']} It was {self.tokenizer.mask_token} ."
            label = item["label"]
        elif self.task_name == "yelp_polarity":
            text = f"{item['text'][:512]} It was {self.tokenizer.mask_token} ."
            label = item["label"]
        elif self.task_name == "ag_news":
            text = f"{item['text'][:512]} This topic is about {self.tokenizer.mask_token} ."
            label = item["label"]
        elif self.task_name == "mrpc":
            text = f"{item['sentence1']} {self.tokenizer.mask_token} , {item['sentence2']}"
            label = item["label"]
        elif self.task_name == "rte":
            text = f"{item['premise']} {self.tokenizer.mask_token} , {item['hypothesis']}"
            label = item["label"]
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        encoding = self.tokenizer(
            text, max_length=self.max_seq_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        mask_pos = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_pos) == 0:
            mask_pos = torch.tensor([self.max_seq_length - 2])
            input_ids[self.max_seq_length - 2] = self.tokenizer.mask_token_id
        mask_pos = mask_pos[0]

        return input_ids, attention_mask, mask_pos, label


def get_verbalizer_ids(task_name: str, tokenizer) -> dict:
    verbalizers = {
        "sst2":          {0: "terrible", 1: "great"},
        "yelp_polarity": {0: "terrible", 1: "great"},
        "ag_news":       {0: "World", 1: "Sports", 2: "Business", 3: "Tech"},
        "mrpc":          {0: "No",  1: "Yes"},
        "rte":           {0: "Yes", 1: "No"},
    }
    return {
        label: tokenizer.encode(f" {word}", add_special_tokens=False)[0]
        for label, word in verbalizers[task_name].items()
    }


# ─────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────

def _compute_f1(all_preds: list, all_labels: list) -> float:
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate_prompt(prompt_model: PromptRoBERTa, z: np.ndarray,
                    dataloader, label_to_token_id: dict, device: str,
                    micro_batch: int = 256, metric: str = "acc"):
    prompt_model.set_prompt(z)
    n_classes = len(label_to_token_id)
    token_ids = torch.tensor([label_to_token_id[i] for i in range(n_classes)], device=device)

    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    for input_ids, attention_mask, mask_positions, labels in dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        mask_positions = mask_positions.to(device)
        labels = labels.to(device)
        n = input_ids.shape[0]

        for start in range(0, n, micro_batch):
            end = min(start + micro_batch, n)
            ml = prompt_model(input_ids[start:end], attention_mask[start:end],
                              mask_positions[start:end])
            cl = ml[:, token_ids]
            total_loss += F.cross_entropy(cl, labels[start:end], reduction="sum").item()
            preds = cl.argmax(dim=-1)
            total_correct += (preds == labels[start:end]).sum().item()
            if metric == "f1":
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels[start:end].cpu().tolist())
        total_samples += n

    avg_loss = total_loss / max(total_samples, 1)
    score = _compute_f1(all_preds, all_labels) if metric == "f1" else total_correct / max(total_samples, 1)
    return avg_loss, score


@torch.no_grad()
def evaluate_batch(prompt_model: PromptRoBERTa, z: np.ndarray,
                   batch, label_to_token_id: dict, device: str,
                   micro_batch: int = 256, metric: str = "acc"):
    prompt_model.set_prompt(z)
    n_classes = len(label_to_token_id)
    token_ids = torch.tensor([label_to_token_id[i] for i in range(n_classes)], device=device)

    input_ids, attention_mask, mask_positions, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    mask_positions = mask_positions.to(device)
    labels = labels.to(device)
    n = input_ids.shape[0]

    total_loss, total_correct = 0.0, 0
    all_preds, all_labels = [], []

    for start in range(0, n, micro_batch):
        end = min(start + micro_batch, n)
        ml = prompt_model(input_ids[start:end], attention_mask[start:end],
                          mask_positions[start:end])
        cl = ml[:, token_ids]
        total_loss += F.cross_entropy(cl, labels[start:end], reduction="sum").item()
        preds = cl.argmax(dim=-1)
        total_correct += (preds == labels[start:end]).sum().item()
        if metric == "f1":
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels[start:end].cpu().tolist())

    avg_loss = total_loss / n
    score = _compute_f1(all_preds, all_labels) if metric == "f1" else total_correct / n
    return avg_loss, score


# ─────────────────────────────────────────────────────────────────
# Multi-GPU parallel evaluation
# ─────────────────────────────────────────────────────────────────

class MultiGPUEvaluator:
    """Evaluate CMA-ES candidates in parallel across multiple GPUs."""

    def __init__(self, model_name: str, n_prompt_tokens: int, intrinsic_dim: int,
                 gpu_ids: list, seed: int = 42):
        self.gpu_ids = gpu_ids
        self.models = {}
        for gid in gpu_ids:
            device = f"cuda:{gid}"
            self.models[gid] = PromptRoBERTa(model_name, n_prompt_tokens, intrinsic_dim,
                                             device, seed)
            print(f"[SGFPT] Loaded model replica on cuda:{gid}")

    def evaluate_candidates(self, solutions, batch, label_to_token_id):
        import concurrent.futures
        n = len(solutions)
        n_gpus = len(self.gpu_ids)
        results = [None] * n

        gpu_assignments = {}
        for i, z in enumerate(solutions):
            gid = self.gpu_ids[i % n_gpus]
            gpu_assignments.setdefault(gid, []).append((i, z))

        def _eval_on_gpu(gid, assignments):
            model = self.models[gid]
            device = f"cuda:{gid}"
            input_ids_tmp, _, _, _ = batch
            seq_len = input_ids_tmp.shape[1] + 50  # input tokens + prompt tokens
            # Budget ~4GB for lm_head output (seq_len x vocab x 4 bytes)
            micro_bs = max(1, min(256, int(4 * 1024**3 / (seq_len * 50265 * 4))))
            n_classes = len(label_to_token_id)
            token_ids = torch.tensor(
                [label_to_token_id[i] for i in range(n_classes)], device=device
            )
            input_ids, attention_mask, mask_positions, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            mask_positions = mask_positions.to(device)
            labels = labels.to(device)
            n = input_ids.shape[0]

            local_results = []
            for global_idx, z in assignments:
                model.set_prompt(z)
                total_loss, total_correct = 0.0, 0
                for start in range(0, n, micro_bs):
                    end = min(start + micro_bs, n)
                    ml = model(input_ids[start:end], attention_mask[start:end],
                               mask_positions[start:end])
                    cl = ml[:, token_ids]
                    total_loss += F.cross_entropy(cl, labels[start:end], reduction="sum").item()
                    total_correct += (cl.argmax(dim=-1) == labels[start:end]).sum().item()
                local_results.append((global_idx, total_loss / n, total_correct / n))
            return local_results

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {pool.submit(_eval_on_gpu, gid, assignments): gid
                       for gid, assignments in gpu_assignments.items()}
            for future in concurrent.futures.as_completed(futures):
                for global_idx, loss, acc in future.result():
                    results[global_idx] = (loss, acc)

        return results

    @property
    def primary_model(self):
        return self.models[self.gpu_ids[0]]


# ─────────────────────────────────────────────────────────────────
# Plaintext training
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_plaintext_training(
    args, cfg: dict, evaluator: MultiGPUEvaluator,
    train_loader, label_to_token_id: dict,
    stop: list, r_per_batch: int,
):
    from algorithm.sep_cma_es import SepCMAES

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_batches = len(train_loader)
    G_total = r_per_batch * n_batches

    cfg_cma = dict(cfg)
    cfg_cma["intrinsic_dim_L"] = int(cfg["intrinsic_dim"])
    cfg_cma["intrinsic_dim_V"] = 0

    opt = SepCMAES(cfg_cma)
    gen_global = 0
    best_loss_overall: float = float("inf")
    best_acc_overall: float = 0.0
    best_prompt: Optional[np.ndarray] = None
    best_acc_prompt: Optional[np.ndarray] = None

    print(f"[SGFPT][plaintext] n_batches={n_batches}  r_per_batch={r_per_batch}  "
          f"G_total={G_total}  gpus={len(evaluator.gpu_ids)}")

    for batch_idx, batch in enumerate(train_loader):
        for ii in range(r_per_batch):
            if stop[0]:
                break
            gen_global += 1
            solutions = opt.ask()

            results = evaluator.evaluate_candidates(solutions, batch, label_to_token_id)

            fitnesses, acc_iter = [], []
            for k, (loss, acc) in enumerate(results):
                fitnesses.append(loss)
                acc_iter.append(acc)
                if loss < best_loss_overall:
                    best_loss_overall = loss
                    best_prompt = np.asarray(solutions[k], dtype=np.float64).copy()
                if acc > best_acc_overall:
                    best_acc_overall = acc
                    best_acc_prompt = np.asarray(solutions[k], dtype=np.float64).copy()

            opt.tell(solutions, fitnesses)
            print(
                f"[SGFPT][plaintext] batch {batch_idx+1}/{n_batches} "
                f"G={gen_global}/{G_total}  sigma={opt.sigma:.4f}  "
                f"best_loss={fitnesses[int(np.argmin(fitnesses))]:.4f}  "
                f"best_acc={acc_iter[int(np.argmax(acc_iter))]*100:.2f}%  "
                f"| overall: loss={best_loss_overall:.4f}  acc={best_acc_overall*100:.2f}%"
            )
        if stop[0]:
            break

    return best_prompt, best_acc_prompt


# ─────────────────────────────────────────────────────────────────
# Per-task runner
# ─────────────────────────────────────────────────────────────────

def run_one_task(
    task_name: str,
    args,
    cfg_base: dict,
    evaluator: MultiGPUEvaluator,
    stop: list,
) -> None:
    """Run training + inference for one task (model already loaded).

    MPC mode order: open port → accept connections → load dataset → train.
    Zero-shot: loaded from existing plaintext results; skipped if not found.
    """
    torch.cuda.empty_cache()

    cfg = dict(cfg_base)
    for k, v in cfg_base.get(task_name, {}).items():
        cfg[k] = v

    log_dir = args.log_dir if args.log_dir else f"log/{task_name}"
    os.makedirs(log_dir, exist_ok=True)
    suffix = "plaintext" if args.plaintext else "mpc"
    prompt_save     = f"{log_dir}/best_prompt_llm_{suffix}.npy"
    prompt_acc_save = f"{log_dir}/best_prompt_llm_{suffix}_acc.npy"
    if not args.tasks:
        if args.prompt_save:     prompt_save     = args.prompt_save
        if args.prompt_acc_save: prompt_acc_save = args.prompt_acc_save

    prompt_model  = evaluator.primary_model
    device        = f"cuda:{evaluator.gpu_ids[0]}"
    intrinsic_dim = int(cfg_base.get("intrinsic_dim", 500))
    eval_metric   = cfg.get("metric", "acc")
    r_per_batch   = int(cfg.get("r_per_batch", 4))

    print(f"\n[SGFPT] {'='*56}")
    print(f"[SGFPT]  Task: {task_name}  metric={eval_metric}  r_per_batch={r_per_batch}")
    print(f"[SGFPT] {'='*56}", flush=True)

    best_prompt: Optional[np.ndarray] = None
    best_acc_prompt: Optional[np.ndarray] = None
    eval_loader = None   # built later; needed for inference

    if not args.inference_only:
        if args.plaintext:
            # Plaintext: load dataset, then train
            train_split, eval_split = load_dataset_splits(task_name)
            max_seq = int(cfg.get("max_seq_length", 128))
            bs      = int(cfg.get("batch_size", 32))
            train_dataset = PromptTuningDataset(train_split, task_name,
                                                prompt_model.tokenizer, max_seq)
            eval_dataset  = PromptTuningDataset(eval_split,  task_name,
                                                prompt_model.tokenizer, max_seq)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
            eval_loader  = torch.utils.data.DataLoader(
                eval_dataset,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
            label_to_token_id = get_verbalizer_ids(task_name, prompt_model.tokenizer)
            print(f"[SGFPT] Train: {len(train_dataset)} samples, {len(train_loader)} batches")
            best_prompt, best_acc_prompt = run_plaintext_training(
                args, cfg, evaluator, train_loader, label_to_token_id, stop, r_per_batch,
            )

        else:
            # ── MPC: open port first, accept connections, THEN load dataset ──
            with PromptServer(host=args.host, port=args.port) as server:
                print(f"[SGFPT] [TASK:{task_name}] Waiting for Party 0 ...", flush=True)
                conn0 = server.accept()
                print(f"[SGFPT] [TASK:{task_name}] Waiting for Party 1 ...", flush=True)
                conn1 = server.accept()
                print(f"[SGFPT] Both parties connected. Loading dataset ...", flush=True)

                # Load dataset after connections are established
                train_split, eval_split = load_dataset_splits(task_name)
                max_seq = int(cfg.get("max_seq_length", 128))
                bs      = args.batch_size if args.batch_size is not None else int(cfg.get("batch_size", 32))
                train_dataset = PromptTuningDataset(train_split, task_name,
                                                    prompt_model.tokenizer, max_seq)
                eval_dataset  = PromptTuningDataset(eval_split,  task_name,
                                                    prompt_model.tokenizer, max_seq)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
                eval_loader  = torch.utils.data.DataLoader(
                    eval_dataset,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
                label_to_token_id = get_verbalizer_ids(task_name, prompt_model.tokenizer)

                n_batches  = len(train_loader)
                G_total    = r_per_batch * n_batches
                batch_list = list(train_loader)
                print(f"[SGFPT] Train: {len(train_dataset)} samples, {n_batches} batches  "
                      f"G_total={G_total}", flush=True)

                best_loss_value = float("inf")
                best_acc_value  = 0.0
                gen_global = 0
                t_comm = 0.0
                t_eval = 0.0
                t_train_start = _time.time()

                with conn0, conn1:
                    for batch_idx, batch in enumerate(batch_list):
                        for ii in range(r_per_batch):
                            if stop[0]:
                                break
                            gen_global += 1

                            # Recv shares (密文)
                            _t0 = _time.time()
                            try:
                                prompts, lambda_, d, bw, scale, party_num = \
                                    recv_two_shares(conn0, conn1)
                            except (ConnectionError, RuntimeError):
                                print(f"[SGFPT] Connection closed at G={gen_global}.")
                                stop[0] = True
                                break
                            t_comm += _time.time() - _t0

                            print(f"[SGFPT] batch {batch_idx+1}/{n_batches} "
                                  f"iter {ii+1}/{r_per_batch} (G={gen_global}/{G_total})")

                            # Evaluate candidates (明文)
                            _t0 = _time.time()
                            results = evaluator.evaluate_candidates(
                                list(prompts), batch, label_to_token_id)
                            t_eval += _time.time() - _t0

                            fitnesses = np.array([r[0] for r in results], dtype=np.float64)
                            accs      = np.array([r[1] for r in results], dtype=np.float64)

                            best_idx = int(np.argmin(fitnesses))
                            if fitnesses[best_idx] < best_loss_value:
                                best_loss_value = fitnesses[best_idx]
                                best_prompt = prompts[best_idx].copy()
                            best_acc_idx = int(np.argmax(accs))
                            if accs[best_acc_idx] > best_acc_value:
                                best_acc_value = accs[best_acc_idx]
                                best_acc_prompt = prompts[best_acc_idx].copy()

                            g = max(gen_global, 1)
                            print(f"[SGFPT] loss=[{fitnesses.min():.4f},{fitnesses.max():.4f}] "
                                  f"acc=[{accs.min():.4f},{accs.max():.4f}] "
                                  f"best_loss={best_loss_value:.4f} best_acc={best_acc_value:.4f}"
                                  f" | comm={t_comm/g:.2f}s eval={t_eval/g:.2f}s")

                            # Simulation: send the same plaintext CE-loss array to both SPs.
                            _t0 = _time.time()
                            conn0.send_fitness(expand_fitness(fitnesses, party_num))
                            conn1.send_fitness(expand_fitness(fitnesses, party_num))
                            t_comm += _time.time() - _t0

                        if stop[0]:
                            break

                t_total = _time.time() - t_train_start
                g = max(gen_global, 1)
                print(f"\n[SGFPT] === TIMING [{task_name}] ===")
                print(f"[SGFPT] Total : {t_total:.1f}s ({t_total/60:.1f}min)")
                print(f"[SGFPT] 密文  : {t_comm:.1f}s ({t_comm/g:.3f}s/gen)")
                print(f"[SGFPT] 明文  : {t_eval:.1f}s ({t_eval/g:.3f}s/gen)")
                print(f"[SGFPT] G     : {gen_global}/{G_total}")
                np.save(f"{log_dir}/timing_{task_name}.npy", {
                    "task": task_name, "total_s": t_total,
                    "comm_s": t_comm, "eval_s": t_eval,
                    "G_done": gen_global, "G_total": G_total,
                    "comm_per_gen_s": t_comm / g, "eval_per_gen_s": t_eval / g,
                })

        if best_prompt is not None:
            np.save(prompt_save, best_prompt)
            print(f"[SGFPT] Loss-best prompt → {prompt_save}")
        if best_acc_prompt is not None:
            np.save(prompt_acc_save, best_acc_prompt)
            print(f"[SGFPT] Acc-best prompt  → {prompt_acc_save}")

    # ── Inference (needs eval_loader) ─────────────────────────────
    # Build eval_loader if not already done (inference_only mode)
    if eval_loader is None:
        _, eval_split = load_dataset_splits(task_name)
        max_seq = int(cfg.get("max_seq_length", 128))
        bs      = int(cfg.get("batch_size", 32))
        eval_dataset = PromptTuningDataset(eval_split, task_name,
                                           prompt_model.tokenizer, max_seq)
        eval_loader  = torch.utils.data.DataLoader(
            eval_dataset, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
        label_to_token_id = get_verbalizer_ids(task_name, prompt_model.tokenizer)

    print(f"\n[SGFPT] === INFERENCE [{task_name}] ===")

    def _run_inference(path: str, label: str):
        try:
            z = np.load(path)
        except FileNotFoundError:
            print(f"[SGFPT] No prompt at {path}, skipping.")
            return None
        loss, score = evaluate_prompt(
            prompt_model, z, eval_loader, label_to_token_id, device, metric=eval_metric)
        print(f"[SGFPT] {label}: loss={loss:.4f}  {eval_metric}={score*100:.2f}%")
        np.save(path.replace(".npy", "_result.npy"),
                {"avg_loss": loss, f"avg_{eval_metric}": score})
        return score

    score_loss = _run_inference(prompt_save, "loss-based")
    score_acc  = _run_inference(prompt_acc_save, "accuracy-based")

    # Compare with zero-shot from plaintext run (don't recompute)
    zs_file = f"{log_dir}/zeroshot_baseline_{task_name}.npy"
    if os.path.exists(zs_file):
        zs_data  = np.load(zs_file, allow_pickle=True).item()
        zs_score = float(zs_data.get(f"avg_{eval_metric}", zs_data.get("avg_acc", 0)))
        ref = score_acc if score_acc is not None else score_loss
        if ref is not None:
            print(f"[SGFPT] Zero-shot (plaintext): {zs_score*100:.2f}%  "
                  f"Improvement: {(ref - zs_score)*100:+.2f}%")

    print(f"[SGFPT] Task {task_name} done.")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Prompt Tuning Server (SGFPT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    task_grp = parser.add_mutually_exclusive_group()
    task_grp.add_argument("--task", default="sst2",
                          choices=["sst2", "yelp_polarity", "ag_news", "mrpc", "rte"],
                          help="Single task (default)")
    task_grp.add_argument("--tasks", default=None,
                          help="Comma-separated task list, e.g. sst2,yelp_polarity,ag_news  "
                               "(model loaded once, tasks run sequentially)")
    parser.add_argument("--config", default=os.path.join(_HERE, "configs", "llm.yaml"))
    parser.add_argument("--gpu", type=int, default=0,
                        help="Primary GPU index (used when --gpus not specified)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids for parallel eval, e.g. '1,2'")
    parser.add_argument("--port", type=int, default=42200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--plaintext", action="store_true",
                        help="Run sep-CMA-ES locally (no TCP / no MPC)")
    parser.add_argument("--skip-zeroshot", action="store_true",
                        help="Skip zero-shot baseline evaluation")
    parser.add_argument("--prompt-save", default=None,
                        help="Override save path for loss-best prompt (single-task only)")
    parser.add_argument("--prompt-acc-save", default=None,
                        help="Override save path for acc-best prompt (single-task only)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training; load saved prompt and run inference only")
    parser.add_argument("--log-dir", default=None,
                        help="Override output directory (default: log/{task})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch_size in config (DataLoader batch size for training)")
    args = parser.parse_args()

    import yaml
    cfg_base = yaml.safe_load(open(args.config))  # raw base config, never mutated

    task_list = [t.strip() for t in args.tasks.split(",")] if args.tasks else [args.task]

    # GPU setup
    gpu_ids = [int(x) for x in args.gpus.split(",")] if args.gpus else [args.gpu]
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[0])
        print(f"[SGFPT] GPUs: {gpu_ids}  ({torch.cuda.get_device_name(gpu_ids[0])} × {len(gpu_ids)})")
    else:
        print("[SGFPT] No CUDA — running on CPU")

    # Load model ONCE
    model_name    = cfg_base.get("backbone", "roberta-large")
    n_prompt      = int(cfg_base.get("n_prompt_tokens", 50))
    intrinsic_dim = int(cfg_base.get("intrinsic_dim", 500))
    seed          = int(cfg_base.get("seed", 42))

    print(f"[SGFPT] Loading {model_name} on GPUs {gpu_ids} "
          f"(prompt_tokens={n_prompt}, d={intrinsic_dim})...")
    evaluator = MultiGPUEvaluator(model_name, n_prompt, intrinsic_dim, gpu_ids, seed)
    print(f"[SGFPT] Model ready.  Tasks to run: {task_list}")

    # Graceful shutdown
    stop = [False]
    def _sigint(*_):
        print("\n[SGFPT] Shutting down...")
        stop[0] = True
    signal.signal(signal.SIGINT, _sigint)

    for task_name in task_list:
        if stop[0]:
            break
        run_one_task(task_name, args, cfg_base, evaluator, stop)

    print("\n[SGFPT] All tasks complete.")


if __name__ == "__main__":
    main()
