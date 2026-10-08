#!/usr/bin/env python3
"""SGFPT data-holder evaluator for the five VLM paper tasks.

--plaintext runs the NumPy Sep-CMA-ES baseline; --inference-only loads a prompt.
The TCP path preserves the original zero-mask research simulation: two C++
processes send duplicate plaintext prompts and receive duplicate plaintext
fitness values. It does not implement the paper's private DH/SP deployment.
See README.md for the protocol gap and exact data flow.
"""

import argparse
import os
import socket
import sys
import signal
import threading
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm

# ── make sure the local B2TPT `clip/` package is importable ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from protocol import PromptServer, PromptConnection, fixed_to_float


# ──────────────────────────────────────────────────────────────────────────────
# Two-party share reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def _recv_share_thread(conn: PromptConnection, result: list, idx: int) -> None:
    """Worker: receive raw u64 share from one connection; store in result[idx]."""
    try:
        result[idx] = conn.recv_raw_share()
    except Exception as e:
        result[idx] = e


def recv_two_shares(conn0: PromptConnection, conn1: PromptConnection):
    """
    Receive both simulation connections concurrently and decode the first copy.
    These are duplicate zero-mask values, not complementary additive shares.

    Returns
    -------
    prompts   : np.ndarray, shape (lambda_, d), dtype float64
    lambda_   : int
    d         : int
    bw        : int
    scale     : int
    party_num : int  – number of data-holding parties (from the header)
    """
    results = [None, None]
    t0 = threading.Thread(target=_recv_share_thread, args=(conn0, results, 0))
    t1 = threading.Thread(target=_recv_share_thread, args=(conn1, results, 1))
    t0.start(); t1.start()
    t0.join();  t1.join()

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            raise RuntimeError(f"Party {i} share receive failed: {r}") from r

    raw0, lambda_, d, bw, scale, party_num  = results[0]
    raw1, lambda1, d1, bw1, scale1, pnum1   = results[1]

    if (lambda1, d1, bw1, scale1, pnum1) != (lambda_, d, bw, scale, party_num):
        raise ValueError(
            f"Party 0 header (lam={lambda_},d={d},bw={bw},scale={scale},pnum={party_num}) "
            f"!= Party 1 header (lam={lambda1},d={d1},bw={bw1},scale={scale1},pnum={pnum1})"
        )

    if not np.array_equal(raw0, raw1):
        raise ValueError("Simulation prompt copies differ between the two parties")
    # setZeroRandomness simulation: masks = 0, so raw0 IS the plaintext.
    # Adding raw1 would double it (raw1 also equals the plaintext when masks=0).
    # A private deployment requires a separate, validated share-export protocol.
    reconstructed = raw0

    prompts = fixed_to_float(reconstructed, bw, scale).reshape(lambda_, d)
    # for i in range(lambda_):
    #     for j in range(d):
    #         print(prompts[i, j], end=" ")
    #     print()
    return prompts, lambda_, d, bw, scale, party_num


def expand_fitness(fitnesses: np.ndarray, party_num: int) -> np.ndarray:
    """
    Simulate data distributed across party_num parties.

    Each candidate's CE loss is split evenly into party_num equal parts so
    that aggregateRowsKernel (which sums party_num values per candidate)
    recovers the original loss by ordinary summation, without FSS share reconstruction.

    Returns float64 array of shape (lambda_ * party_num,) in row-major order:
        [ loss[0]/p, loss[0]/p, ...,  loss[1]/p, loss[1]/p, ..., ]
    """
    part = fitnesses / party_num          # shape (lambda_,)
    return np.repeat(part, party_num)     # shape (lambda_ * party_num,)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_zeroshot_clip(prompt_clip, eval_loader, device: str):
    """
    Evaluate pure zero-shot CLIP (no learned prompts) on an eval set.
    
    Uses the standard template: "A photo of a {class}." for all classes.
    This establishes a true baseline without any prompt optimization.
    
    Returns
    -------
    avg_loss : float
    avg_acc  : float
    """
    import clip
    import torch.nn.functional as F
    
    # Build zero-shot text prompts
    temp_p = "A photo of a {}."
    prompts_p = [temp_p.format(c.replace("_", " ")) for c in prompt_clip.classes]
    prompts_p = torch.cat([clip.tokenize(p) for p in prompts_p]).to(device)
    
    # Encode text features using original CLIP (no learned prompts)
    with torch.no_grad():
        text_features_p = prompt_clip.model.encode_text(prompts_p)
        text_features_p = text_features_p / text_features_p.norm(dim=-1, keepdim=True)
    
    all_losses = []
    all_accs = []
    
    for batch_idx, batch in enumerate(eval_loader):
        image, label = prompt_clip.parse_batch(batch)
        
        # Encode images using original CLIP (no learned prompts)
        image_features = prompt_clip.image_encoder_clip(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logit_scale = prompt_clip.logit_scale.exp()
        logits = logit_scale * image_features @ text_features_p.t()
        
        # Loss and accuracy
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fn(logits, label)
        
        prediction = logits.argmax(dim=-1)
        correct = (prediction == label).float().sum()
        acc = correct / int(label.shape[0])
        
        all_losses.append(loss.item())
        all_accs.append(acc.item())
    
    return float(np.mean(all_losses)), float(np.mean(all_accs))


@torch.no_grad()
def eval_candidates_full(prompt_clip, prompt_vectors: np.ndarray,
                         train_loader, intrinsic_dim_L: int, device: str):
    """
    Evaluate lambda prompt vectors over the FULL train_loader.

    Each candidate is evaluated on every batch and the losses/accuracies
    are averaged across batches, matching the original B2TPT evaluation
    loop where prompt_clip.eval(prompt, batch, ii, r) is called.

    Parameters
    ----------
    prompt_clip     : PromptCLIP_Shallow instance
    prompt_vectors  : float64 numpy array (lambda, d)
    train_loader    : DataLoader for the full train set
    intrinsic_dim_L : split index between text and vision intrinsic dims

    Returns
    -------
    fitnesses : numpy (lambda,)  – mean CE loss  (lower = better)
    accs      : numpy (lambda,)  – mean accuracy (higher = better)
    """
    text_projs  = prompt_clip.generate_text_prompts(
        [v[:intrinsic_dim_L] for v in prompt_vectors])
    image_projs = prompt_clip.generate_visual_prompts(
        [v[intrinsic_dim_L:] for v in prompt_vectors])

    n_batches = len(train_loader)
    # accumulate per-candidate totals across all batches
    loss_sum = np.zeros(len(prompt_vectors), dtype=np.float64)
    acc_sum  = np.zeros(len(prompt_vectors), dtype=np.float64)

    for ii, batch in enumerate(train_loader):
        r = n_batches
        for j, (pt, pv) in enumerate(zip(text_projs, image_projs)):
            loss, acc = prompt_clip.eval((pt, pv), batch, ii=ii, r=r)
            loss_sum[j] += loss
            acc_sum[j]  += acc

    return (loss_sum / n_batches).astype(np.float64), \
           (acc_sum  / n_batches).astype(np.float64)


def dry_run_eval(lambda_: int) -> np.ndarray:
    """Return random fitness values – used when --dry-run is active."""
    fitness = np.random.rand(lambda_).astype(np.float64)
    print(f"[SGFPT][dry-run] Returning random fitness: "
          f"min={fitness.min():.4f}  max={fitness.max():.4f}")
    return fitness


@torch.no_grad()
def run_plaintext_training(
    args,
    cfg: dict,
    prompt_clip,
    train_loader,
    intrinsic_L: int,
    intrinsic_V: int,
    stop: list,
    r_per_batch: int,
):
    """
    Local sep-CMA-ES prompt tuning (no TCP / no MPC).
    Matches B2TPT.py's inner loop: for each train batch, run r_per_batch
    CMA generations evaluated on that batch only.

    Returns
    -------
    best_prompt      : np.ndarray or None  – intrinsic vector with lowest CE loss
    best_acc_prompt  : np.ndarray or None  – intrinsic vector with highest accuracy
    """
    from algorithm.sep_cma_es import SepCMAES

    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_pop     = int(cfg["popsize"])
    n_batches = len(train_loader)
    G_total   = r_per_batch * n_batches

    opt = SepCMAES(cfg)
    gen_global = 0

    best_loss_overall: float            = float('inf')
    best_acc_overall:  float            = 0.0
    best_prompt:       Optional[np.ndarray] = None
    best_acc_prompt:   Optional[np.ndarray] = None

    print("[SGFPT][plaintext] Running local sep-CMA-ES (no socket; matches B2TPT.py loop).")

    for batch_idx, batch in enumerate(train_loader):
        for ii in range(r_per_batch):
            if stop[0]:
                break
            if args.rounds is not None and gen_global >= args.rounds:
                break
            gen_global += 1

            solutions         = opt.ask()
            prompt_text_list  = prompt_clip.generate_text_prompts(
                [x[:intrinsic_L] for x in solutions])
            prompt_image_list = prompt_clip.generate_visual_prompts(
                [x[intrinsic_L:] for x in solutions])
            results = [
                prompt_clip.eval(x, batch, ii, r_per_batch)
                for x in zip(prompt_text_list, prompt_image_list)
            ]

            fitnesses = []
            acc_iter  = []
            for k in range(n_pop):
                fit, acc = float(results[k][0]), float(results[k][1])
                fitnesses.append(fit)
                acc_iter.append(acc)
                if fit < best_loss_overall:
                    best_loss_overall = fit
                    best_prompt = np.asarray(solutions[k], dtype=np.float64).copy()
                if acc > best_acc_overall:
                    best_acc_overall = acc
                    best_acc_prompt  = np.asarray(solutions[k], dtype=np.float64).copy()

            opt.tell(solutions, fitnesses)

            cur_loss_idx = int(np.argmin(fitnesses))
            cur_acc_idx  = int(np.argmax(acc_iter))
            print(
                f"[SGFPT][plaintext] batch {batch_idx+1}/{n_batches} "
                f"cma-iter {ii+1}/{r_per_batch} (G={gen_global}/{G_total}) "
                f"sigma={opt.sigma:.4f} "
                f"best_loss={fitnesses[cur_loss_idx]:.4f} "
                f"best_acc={acc_iter[cur_acc_idx]*100:.2f}% "
                f"| overall: loss={best_loss_overall:.4f} acc={best_acc_overall*100:.2f}%"
            )

        if stop[0]:
            break
        if args.rounds is not None and gen_global >= args.rounds:
            break

    return best_prompt, best_acc_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt_clip(task_name: str, cfg: dict, device: str):
    """Load CLIP backbone and initialise PromptCLIP_Shallow."""
    import clip as clip_pkg
    from model.prompt_clip import PromptCLIP_Shallow

    _, preprocess = clip_pkg.load(cfg["backbone"], device=device)
    data_root = cfg.get("data_dir", "./data")

    if task_name == "cifar100":
        from dataset.cifar100 import load_train_cifar100, load_test_cifar100
        train_data, train_loader = load_train_cifar100(
            batch_size=cfg["batch_size"], preprocess=preprocess, root=data_root)
        test_data, test_loader = load_test_cifar100(
            batch_size=cfg["batch_size"], preprocess=preprocess, root=data_root,
            shuffle=cfg.get("shuffle_test", True))
    else:
        import importlib
        modules = {"dtd": "dtd", "eurosat": "eurosat",
                   "flower102": "flowers102", "pets": "pets"}
        if task_name not in modules:
            raise ValueError(f"Unsupported paper task: {task_name}")
        name = modules[task_name]
        module = importlib.import_module(f"dataset.{name}")
        train_data, train_loader = getattr(module, f"load_{name}_train")(
            batch_size=cfg["batch_size"], preprocess=preprocess, root=data_root)
        test_data, test_loader = getattr(module, f"load_{name}_test")(
            batch_size=cfg["batch_size"], preprocess=preprocess, root=data_root,
            shuffle=cfg.get("shuffle_test", True))
    classes = train_data.classes
    n_cls = len(classes)
    print(f"[SGFPT] {task_name}: {n_cls} classes, {len(train_loader)} train batches")

    # Build train/test loaders (train used for optimization; test used for reporting)
    prompt_clip = PromptCLIP_Shallow(task_name, cfg, classes, n_cls)
    prompt_clip.text_encoder.set_context(prompt_clip.get_text_information())
    prompt_clip.image_encoder.set_context(prompt_clip.get_image_information())
    return prompt_clip, train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SGFPT VLM evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",    default="dtd",       help="Dataset task name")
    parser.add_argument("--port",    type=int, default=42200, help="TCP listen port")
    parser.add_argument("--host",    default="127.0.0.1",   help="Bind address")
    parser.add_argument("--rounds",  type=int, default=None,
                        help="Stop after N rounds (None = run forever)")
    parser.add_argument("--config",  default=os.path.join(_HERE, "configs", "vlm.yaml"))
    parser.add_argument("--gpu",     type=int, default=0,
                        help="CUDA device index (e.g. 0, 1, 2, 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip real CLIP evaluation; return random fitness "
                             "(useful for testing the communication layer)")
    # dry-run only: expected dimensions for validation
    parser.add_argument("--lambda",  dest="lam", type=int, default=30,
                        help="[dry-run] expected number of candidates")
    parser.add_argument("--dim",     type=int,   default=400,
                        help="[dry-run] expected intrinsic dimension")
    parser.add_argument("--prompt-save", default="best_prompt.npy",
                        help="Path to save/load the loss-based best prompt after training")
    parser.add_argument("--prompt-acc-save", default=None,
                        help="Path to save the accuracy-based best prompt "
                             "(defaults to <prompt-save stem>_acc.npy)")
    parser.add_argument("--inference-only", action="store_true",
                        help="Skip training; load saved prompt and run inference only")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Subsample each training batch to this many samples before evaluation "
                             "(default: use full batch from DataLoader)")
    parser.add_argument("--log-dir", default=None,
                        help="Override output directory for results (default: log/{task})")
    parser.add_argument("--data-dir", default=None,
                        help="Override data root directory (default: ./data from config)")
    parser.add_argument(
        "--plaintext",
        action="store_true",
        help="Run sep-CMA-ES + CLIP fully locally (no TCP / no MPC). "
             "Matches B2TPT.py loop for comparison against the MPC path.",
    )
    args = parser.parse_args()

    if args.plaintext and args.dry_run:
        print("[SGFPT] --plaintext cannot be combined with --dry-run.")
        sys.exit(1)

    # Config key normalization: config uses singular `flower102`.
    if args.task == "flowers102":
        args.task = "flower102"

    # ── load config ──────────────────────────────────────────────────────────
    import yaml
    cfg = yaml.safe_load(open(args.config))
    for k, v in cfg.get(args.task, {}).items():
        cfg[k] = v
    cfg.setdefault("opt_name",   "shallow_cma")
    cfg.setdefault("backbone",   "ViT-B/16")
    cfg.setdefault("data_dir",   "./data")
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    cfg.setdefault("output_dir", "./result")
    cfg.setdefault("parallel",   False)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if args.gpu < 0 or args.gpu >= n_gpus:
            raise ValueError(f"--gpu {args.gpu} is out of range; "
                             f"available devices: 0..{n_gpus-1}")
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
        print(f"[SGFPT] Device : {device}  ({torch.cuda.get_device_name(args.gpu)})")
        print(f"[SGFPT] PyTorch CUDA : {torch.version.cuda}")
    else:
        device = "cpu"
        print("[SGFPT] Device : cpu (no CUDA available)")

    # ── optionally load real model ────────────────────────────────────────────
    prompt_clip  = None
    train_loader = None
    test_loader = None
    intrinsic_L  = cfg.get("intrinsic_dim_L", 200)
    intrinsic_V  = cfg.get("intrinsic_dim_V", 200)
    expected_dim = intrinsic_L + intrinsic_V

    np.random.seed(int(cfg.get("seed", 42)))
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))

    if not args.dry_run:
        print(f"[SGFPT] Loading CLIP model ({cfg['backbone']})...")
        prompt_clip, train_loader, test_loader = build_prompt_clip(args.task, cfg, device)
    else:
        print("[SGFPT] *** DRY-RUN mode: skipping CLIP/dataset load ***")
        expected_dim = args.dim

    # ── derive acc-save path if not specified ─────────────────────────────────
    if args.prompt_acc_save is None:
        stem, ext = os.path.splitext(args.prompt_save)
        args.prompt_acc_save = stem + "_acc" + (ext or ".npy")

    # ── graceful shutdown on Ctrl-C ───────────────────────────────────────────
    stop = [False]
    def _sigint(*_):
        print("\n[SGFPT] Shutting down...")
        stop[0] = True
    signal.signal(signal.SIGINT, _sigint)

    # ══════════════════════════════════════════════════════════════════════════
    # ZERO-SHOT BASELINE INFERENCE (Before Training)
    #   Run pure zero-shot CLIP inference (no learned prompts) on all test
    #   batches to establish a true baseline for comparison.
    # ══════════════════════════════════════════════════════════════════════════
    zeroshot_results = {}  # store per-batch results
    if not args.dry_run and not getattr(args, "inference_only", False):
        print(f"\n[SGFPT] === ZERO-SHOT CLIP BASELINE (No Learned Prompts) ===")
        print(f"[SGFPT] Using template: 'A photo of a {{class}}.'")
        
        avg_zeroshot_loss, avg_zeroshot_acc = eval_zeroshot_clip(prompt_clip, test_loader, device)
        
        print(f"[SGFPT] Zero-shot CLIP baseline:")
        print(f"[SGFPT]   avg_loss = {avg_zeroshot_loss:.4f}")
        print(f"[SGFPT]   avg_acc  = {avg_zeroshot_acc*100:.2f}%")
        
        # Save zero-shot results to file
        zeroshot_file = f"zeroshot_baseline_{args.task}.npy"
        np.save(zeroshot_file, {
            'avg_loss': avg_zeroshot_loss,
            'avg_acc': avg_zeroshot_acc,
            'method': 'pure_zeroshot_clip',
            'template': 'A photo of a {class}.'
        })
        print(f"[SGFPT] Zero-shot results saved to {zeroshot_file}")
        
        # Note: We don't store per-batch results for zero-shot since
        # the training loop evaluates on single batches with learned prompts,
        # which is not directly comparable to full-dataset zero-shot.

    # ══════════════════════════════════════════════════════════════════════════
    # TRAINING PHASE
    #   G = 4 × len(train_loader) total MPC rounds.
    #   For each train batch we serve r=4 CMA generations; eval is done only
    #   on the current batch (matching B2TPT.py's inner loop exactly).
    #   After all batches the best prompt (lowest-loss candidate of the last
    #   generation) is saved to disk.
    # ══════════════════════════════════════════════════════════════════════════
    best_prompt = None   # shape (d,), float64 – filled during training (best loss)
    best_acc_prompt = None  # shape (d,), float64 – prompt with best accuracy
    best_loss_value = float('inf')  # track overall best loss
    best_acc_value = 0.0  # track overall best accuracy
    
    if not getattr(args, "inference_only", False):
        # Read r_per_batch (evaluation_time) from task-specific config, default to 4
        task_cfg = cfg.get(args.task, {})
        r_per_batch = task_cfg.get("r_per_batch", 4)
        n_batches   = len(train_loader) if train_loader is not None else 1
        G_total     = r_per_batch * n_batches
        print(f"\n[SGFPT] === TRAINING PHASE ===")
        print(f"[SGFPT] Dataset: {args.task}")
        print(f"[SGFPT] train batches={n_batches}  r_per_batch={r_per_batch}  "
              f"total evolutions G={G_total} (target: ≤300)")

        if args.plaintext:
            # ── Plaintext sep-CMA-ES (no TCP, no MPC) ────────────────────────
            best_prompt, best_acc_prompt = run_plaintext_training(
                args, cfg, prompt_clip, train_loader,
                intrinsic_L, intrinsic_V, stop, r_per_batch,
            )
        else:
            # ── MPC path via TCP ──────────────────────────────────────────────
            batch_list = list(train_loader) if train_loader is not None else [None] * n_batches
            gen_global = 0

            with PromptServer(host=args.host, port=args.port) as server:
                print(f"[SGFPT] Listening on {args.host}:{args.port} ...")

                print("[SGFPT] Waiting for Party 0 ...")
                conn0 = server.accept()
                print("[SGFPT] Waiting for Party 1 ...")
                conn1 = server.accept()
                print("[SGFPT] Both parties connected. Starting training loop.")

                with conn0, conn1:
                    for batch_idx, batch in enumerate(batch_list):
                        # Subsample batch if --eval-batch-size is set
                        if args.eval_batch_size is not None and batch is not None:
                            if isinstance(batch, dict):
                                n_avail = batch["image"].shape[0]
                                if args.eval_batch_size < n_avail:
                                    idx = torch.randperm(n_avail)[:args.eval_batch_size]
                                    batch = {k: v[idx] if isinstance(v, torch.Tensor) else v
                                             for k, v in batch.items()}
                            else:
                                imgs, lbls = batch
                                n_avail = imgs.shape[0]
                                if args.eval_batch_size < n_avail:
                                    idx = torch.randperm(n_avail)[:args.eval_batch_size]
                                    batch = (imgs[idx], lbls[idx])

                        for ii in range(r_per_batch):
                            if stop[0] or (args.rounds is not None and gen_global >= args.rounds):
                                stop[0] = True
                                break
                            gen_global += 1
                            try:
                                prompts, lambda_, d, bw, scale, party_num = \
                                    recv_two_shares(conn0, conn1)
                            except (ConnectionError, RuntimeError):
                                print(f"[SGFPT] Clients finished / connection closed at gen {gen_global}.")
                                stop[0] = True
                                break

                            print(f"[SGFPT] batch {batch_idx+1}/{n_batches} "
                                  f"cma-iter {ii+1}/{r_per_batch} "
                                  f"(G={gen_global}/{G_total})  "
                                  f"λ={lambda_} d={d}  "
                                  f"prompt mean={prompts.mean():.4f}")

                            try:
                                if not args.dry_run:
                                    if d != expected_dim:
                                        raise ValueError(f"d={d} != expected {expected_dim}")
                                    text_projs  = prompt_clip.generate_text_prompts(
                                        [v[:intrinsic_L] for v in prompts])
                                    image_projs = prompt_clip.generate_visual_prompts(
                                        [v[intrinsic_L:] for v in prompts])
                                    results = [prompt_clip.eval((pt, pv), batch, ii=ii, r=r_per_batch)
                                               for pt, pv in zip(text_projs, image_projs)]
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

                                    print(f"[SGFPT] loss=[{fitnesses.min():.4f},{fitnesses.max():.4f}] "
                                          f"acc=[{accs.min():.4f},{accs.max():.4f}] "
                                          f"best_loss={fitnesses[best_idx]:.4f} "
                                          f"best_acc={accs[best_idx]:.4f} | "
                                          f"overall_best: loss={best_loss_value:.4f} acc={best_acc_value:.4f}")
                                else:
                                    fitnesses = dry_run_eval(lambda_)

                                expanded = expand_fitness(fitnesses, party_num)
                                conn0.send_fitness(expanded)
                                conn1.send_fitness(expanded)

                            except Exception as exc:
                                import traceback
                                traceback.print_exc()
                                stop[0] = True
                                break

                        if stop[0]:
                            break

        # Save best prompts
        if best_prompt is not None:
            np.save(args.prompt_save, best_prompt)
            print(f"[SGFPT] Loss-based best prompt saved to {args.prompt_save}  "
                  f"(shape={best_prompt.shape})")
        if best_acc_prompt is not None:
            np.save(args.prompt_acc_save, best_acc_prompt)
            print(f"[SGFPT] Accuracy-based best prompt saved to {args.prompt_acc_save}  "
                  f"(shape={best_acc_prompt.shape})")

    # ══════════════════════════════════════════════════════════════════════════
    # INFERENCE PHASE
    #   Load the saved best prompt and run one forward pass per test batch.
    #   No MPC communication needed – the prompt is now plaintext.
    #   Can optionally test both loss-based and accuracy-based prompts.
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[SGFPT] === INFERENCE PHASE ===")
    if args.dry_run:
        print("[SGFPT] dry-run: skipping inference.")
        return

    def _run_inference(prompt_path: str, label: str):
        """Load a saved prompt vector and evaluate on the full test set."""
        try:
            p = np.load(prompt_path)
        except FileNotFoundError:
            print(f"[SGFPT] No saved prompt found at {prompt_path}. Skipping.")
            return None
        print(f"[SGFPT] Loaded {label} prompt from {prompt_path}")
        t_proj = prompt_clip.generate_text_prompts([p[:intrinsic_L]])[0]
        v_proj = prompt_clip.generate_visual_prompts([p[intrinsic_L:]])[0]
        accs = []
        n = len(test_loader)
        for bi, batch in enumerate(test_loader):
            loss, acc = prompt_clip.eval((t_proj, v_proj), batch, ii=bi, r=n)
            accs.append(acc)
            print(f"[SGFPT] inference batch {bi+1}/{n}  acc={acc*100:.2f}%  loss={loss:.4f}")
        avg = float(np.mean(accs))
        print(f"\n[SGFPT] {label.capitalize()} prompt - Average accuracy: {avg*100:.2f}%")
        result_path = prompt_path.replace(".npy", "_result.npy")
        np.save(result_path, np.array(accs))
        print(f"[SGFPT] Result saved to {result_path}")
        return avg

    print(f"\n[SGFPT] --- Testing loss-based prompt ---")
    avg_loss_acc = _run_inference(args.prompt_save, "loss-based")

    if args.prompt_acc_save != args.prompt_save:
        print(f"\n[SGFPT] --- Testing accuracy-based prompt ---")
        avg_acc_acc = _run_inference(args.prompt_acc_save, "accuracy-based")

    # Compare with zero-shot baseline saved during this run
    zeroshot_file = f"zeroshot_baseline_{args.task}.npy"
    if os.path.exists(zeroshot_file):
        zs_data = np.load(zeroshot_file, allow_pickle=True).item()
        zs_avg_acc = float(zs_data['avg_acc'])
        ref_acc = avg_acc_acc if (args.prompt_acc_save != args.prompt_save
                                  and avg_acc_acc is not None) else avg_loss_acc
        if ref_acc is not None:
            print(f"[SGFPT] Zero-shot baseline accuracy: {zs_avg_acc*100:.2f}%")
            print(f"[SGFPT] Improvement over zero-shot: {(ref_acc - zs_avg_acc)*100:+.2f}%")

    print("[SGFPT] Server closed.")


if __name__ == "__main__":
    main()
