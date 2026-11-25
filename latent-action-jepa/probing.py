import os
os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf
import torch

from evaluate_next_state import (
    available_evaluation_runs,
     evaluation_checkpoint_run,
	 evaluation_overwrite,
    evaluation_run_filter,
     load_evaluation_model,
    make_validation_loader,
	 per_run_metrics_path,
)
from stable_worldmodel.wm.utils import load_pretrained
from sweep import SEEDS

# --------------------------------------------------------------------------------------
# Linear probe (closed-form ridge+ r2_score).
# --------------------------------------------------------------------------------------

def evaluation_seeds():
   configured = os.environ.get("EVAL_SEEDS")
   if configured is None:
       return SEEDS
   try:
       seeds = tuple(int(seed.strip()) for seed in configured.split(",") if seed.strip())
   except ValueError as err:
       raise ValueError(f"EVAL_SEEDS must be comma-separated integers, got {configured!r}") from err
   if not seeds:
       raise ValueError("EVAL_SEEDS must contain at least one seed.")
   unknown = set(seeds) - set(SEEDS)
   if unknown:
       raise ValueError(f"EVAL_SEEDS contains seeds outside the sweep: {sorted(unknown)}")
   return seeds

def ridge_fit_predict(x_tr, y_tr, x_te, alpha=1.0):
   x_tr, y_tr, x_te = (t.double() for t in (x_tr, y_tr, x_te))
   x_mean = x_tr.mean(0, keepdim=True)
   y_mean = y_tr.mean(0, keepdim=True)
   xc, yc = x_tr - x_mean, y_tr - y_mean
   d = xc.size(1)
   gram = xc.t() @ xc + alpha * torch.eye(d, dtype=xc.dtype)
   weights = torch.linalg.solve(gram, xc.t() @ yc)  # (d, out)
   return (x_te - x_mean) @ weights + y_mean


def r2_score(y_true, y_pred):
   y_true, y_pred = y_true.double(), y_pred.double()
   ss_res = (y_true - y_pred).pow(2).sum(0)
   ss_tot = (y_true - y_true.mean(0, keepdim=True)).pow(2).sum(0)
   r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
   return r2.mean().item()


def mlp_probe(x_tr, y_tr, x_te, y_te, hidden=256, epochs=300, lr=1e-3, seed=0):
   torch.manual_seed(seed)
   x_tr, y_tr, x_te, y_te = (t.float() for t in (x_tr, y_tr, x_te, y_te))
   xm, xs = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
   ym, ys = y_tr.mean(0, keepdim=True), y_tr.std(0, keepdim=True).clamp_min(1e-6)
   x_tr, y_tr = (x_tr - xm) / xs, (y_tr - ym) / ys
   x_te, y_te = (x_te - xm) / xs, (y_te - ym) / ys


   net = torch.nn.Sequential(
       torch.nn.Linear(x_tr.size(1), hidden), torch.nn.GELU(),
       torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
       torch.nn.Linear(hidden, y_tr.size(1)),
   )
   opt = torch.optim.Adam(net.parameters(), lr=lr)
   for _ in range(epochs):  # full-batch — the extracted feature set is small
       opt.zero_grad()
       torch.nn.functional.mse_loss(net(x_tr), y_tr).backward()
       opt.step()
   net.eval()
   with torch.no_grad():
       return r2_score(y_te, net(x_te))

def per_dim_variance(feats):
   return feats.var(dim=0, unbiased=False).mean().item()


def effective_rank(feats):
   feats = (feats - feats.mean(0, keepdim=True)).double()
   cov = (feats.t() @ feats) / max(feats.size(0) - 1, 1)
   eig = torch.linalg.eigvalsh(cov).clamp_min(0)
   p = eig / eig.sum().clamp_min(1e-12)
   entropy = -(p * (p + 1e-12).log()).sum()
   return entropy.exp().item()


def adaln_param_norm(model):
   total = 0.0
   for m in model.modules():
       mod = getattr(m, "adaLN_modulation", None)
       if mod is None:
           continue
       last = mod[-1]  # the zero-initialized Linear
       total += last.weight.detach().float().pow(2).sum().item()
       if last.bias is not None:
           total += last.bias.detach().float().pow(2).sum().item()
   return total**0.5

def _is_code_model(model):
   return getattr(model, "conditioning", "action") == "code" and getattr(model, "idm", None) is not None


@torch.no_grad()
def _conditioning(model, emb, batch, device):
   if _is_code_model(model):
       cond_emb, _ = model.infer_code(emb)  # (B, T-1, A)
       return cond_emb
   act = torch.nan_to_num(batch["action"].to(device), 0.0)
   return model.action_encoder(act)  # (B, T, A)


@torch.no_grad()
def shuffle_diagnostics(model, loader, ctx_len, n_preds, device, num_batches=None):
   sums = {"normal": 0.0, "shuf_z": 0.0, "shuf_c": 0.0, "nocode": 0.0}
   count = 0
   for i, batch in enumerate(loader):
       if num_batches is not None and i >= num_batches:
           break
       pixels = batch["pixels"].float().to(device)
       emb = model.encode({"pixels": pixels})["emb"]  # (B, T, D)
       cond = _conditioning(model, emb, batch, device)


       ctx_emb = emb[:, :ctx_len]          # (B, ctx_len, D)
       cond = cond[:, :ctx_len]            # (B, ctx_len, A)
       tgt = emb[:, n_preds:]              # (B, ctx_len, D) — the frames to predict
       perm = torch.randperm(emb.size(0), device=device)


       def mse(pred):
           return (pred - tgt).pow(2).mean().item()


       sums["normal"] += mse(model.predict(ctx_emb, cond))
       sums["shuf_z"] += mse(model.predict(ctx_emb[perm], cond))
       sums["shuf_c"] += mse(model.predict(ctx_emb, cond[perm]))
       sums["nocode"] += mse(model.predict(ctx_emb, torch.zeros_like(cond)))
       count += 1


   means = {k: v / max(count, 1) for k, v in sums.items()}
   means["cheat_ratio"] = means["shuf_z"] / max(means["normal"], 1e-12)
   means["dead_ratio"] = means["shuf_c"] / max(means["normal"], 1e-12)
   return means


def _probe_pair(x_tr, y_tr, x_te, y_te, alpha=1.0):
   return {
       "ridge": r2_score(y_te, ridge_fit_predict(x_tr, y_tr, x_te, alpha=alpha)),
       "mlp": mlp_probe(x_tr, y_tr, x_te, y_te),
   }


def _onehot(token, vocab):
   return torch.nn.functional.one_hot(token.long(), num_classes=vocab).float()


@torch.no_grad()
def extract_probe_data(model, loader, device, num_batches=None, state_key="state"):
    is_code = _is_code_model(model)
    has_quant = is_code and getattr(model, "quantizer", None) is not None
    buf = {k: [] for k in ("z_t", "code", "token", "delta", "action", "state", "state_delta")}

    for i, batch in enumerate(loader):
       if num_batches is not None and i >= num_batches:
           break
       pixels = batch["pixels"].float().to(device)
       emb = model.encode({"pixels": pixels})["emb"]   # (B, T, D)
       z_t, z_tp1 = emb[:, :-1], emb[:, 1:]            # (B, T-1, D)
       T1 = z_t.size(1)


       buf["z_t"].append(z_t.reshape(-1, z_t.size(-1)).cpu())
       buf["delta"].append((z_tp1 - z_t).reshape(-1, z_t.size(-1)).cpu())


       if is_code:
           code = model.idm(z_t, z_tp1)                # (B, T-1, Dc)
           buf["code"].append(code.reshape(-1, code.size(-1)).cpu())
           if has_quant and hasattr(model.quantizer, "to_token"):
               buf["token"].append(model.quantizer.to_token(code).reshape(-1).cpu())


       act = batch["action"][:, :T1].to(device).float()
       buf["action"].append(act.reshape(-1, act.size(-1)).cpu())
       state = batch[state_key].float()
       if state.size(1) < T1 + 1 or state.size(-1) < 5:
           raise ValueError(
               f"Expected {state_key!r} to contain at least {T1 + 1} frames, agent position, "
               f"block position, and block orientation at indices (0, 1), (2, 3), and 4; "
               f"got shape {tuple(state.shape)}."
           )
       state_t = state[:, :T1]
       state_delta = state[:, 1:T1 + 1] - state_t
       buf["state"].append(state_t.reshape(-1, state_t.size(-1)).cpu())
       buf["state_delta"].append(state_delta.reshape(-1, state_delta.size(-1)).cpu())


    data = {k: (torch.cat(v) if v else None) for k, v in buf.items()}
    keep = ~torch.isnan(data["action"]).any(dim=-1)
    keep &= ~torch.isnan(data["state"]).any(dim=-1)
    return {k: (t[keep] if t is not None else None) for k, t in data.items()}


def _train_test_indices(num_samples, train_fraction, seed):
   if not 0.0 < train_fraction < 1.0:
       raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}.")
   if num_samples < 2:
       raise ValueError("At least two valid state embeddings are required for probing.")

   generator = torch.Generator().manual_seed(seed)
   permutation = torch.randperm(num_samples, generator=generator)
   train_size = min(max(int(num_samples * train_fraction), 1), num_samples - 1)
   return permutation[:train_size], permutation[train_size:]


def action_probe_metrics(data, train_fraction=0.8, alpha=1.0, seed=0):
   action = data["action"]
   sources = {"z_t": data["z_t"], "delta": data["delta"]}
   if data["code"] is not None:
       sources["code"] = data["code"]
       # the conditional test. a code may encode the action state-dependently,
       # which scores poorly alone while still adding information given z_t --
       # so the gain below is measured against this, not against code alone.
       sources["z_t_and_code"] = torch.cat([data["z_t"], data["code"]], dim=-1)

   train_indices, test_indices = _train_test_indices(
       action.size(0), train_fraction, seed
   )
   metrics = {
       "num_train_samples": len(train_indices),
       "num_test_samples": len(test_indices),
       "action_dim": action.size(-1),
   }
   for name, features in sources.items():
       for probe_name, value in _probe_pair(
           features[train_indices], action[train_indices],
           features[test_indices], action[test_indices], alpha=alpha,
       ).items():
           metrics[f"{name}_to_action_{probe_name}_r2"] = value
   if "code" in sources:
       for probe_name in ("ridge", "mlp"):
           state_r2 = metrics[f"z_t_to_action_{probe_name}_r2"]
           metrics[f"code_gain_over_state_{probe_name}"] = (
               metrics[f"z_t_and_code_to_action_{probe_name}_r2"] - state_r2
           )
           metrics[f"state_headroom_{probe_name}"] = 1.0 - state_r2
   return metrics


def position_probe_metrics(features, state, state_delta, train_fraction=0.8, alpha=1.0, seed=0):
   train_indices, test_indices = _train_test_indices(
       features.size(0), train_fraction, seed
   )
   train_size = len(train_indices)
   targets = {
       "block_orientation": state[:, 4:5],
         "agent_displacement": state_delta[:, (0, 1)],
         "block_displacement": state_delta[:, (2, 3)],
   }
   metrics = {"num_train_samples": train_size, "num_test_samples": len(test_indices)}
   for name, target in targets.items():
       for probe_name, value in _probe_pair(
           features[train_indices], target[train_indices],
           features[test_indices], target[test_indices], alpha=alpha,
       ).items():
           metrics[f"{name}_{probe_name}_r2"] = value
   return metrics


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    cache_dir = os.environ["STABLEWM_HOME"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    action_results = []

    def show(name, kind, metrics):
        formatted = " ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in metrics.items()
        )
        print(f"[{kind}] {name} {formatted}", flush=True)
    selected_runs = evaluation_run_filter()
    available = list(
        available_evaluation_runs(
            cache_dir, cfg.eval_epoch, evaluation_seeds(), selected_runs=selected_runs
        )
    )
    if selected_runs is not None:
        known_runs = {name for _, _, name, _ in available}
        unknown_runs = selected_runs - known_runs
        if unknown_runs:
            print(f"[skip] unavailable requested runs: {sorted(unknown_runs)}", flush=True)
    if not available:
        raise FileNotFoundError(
            f"No checkpoints found for SWEEP_TAG={os.environ.get('SWEEP_TAG')!r} "
            f"at epoch {cfg.eval_epoch}."
        )
    for arm, seed, name, config_path in available:
        position_path = (
            per_run_metrics_path(cfg.eval_output_dir, "position_probe_metrics", name)
            if cfg.eval_output_dir is not None
            else None
        )
        action_path = (
            per_run_metrics_path(cfg.eval_output_dir, "action_probe_metrics", name)
            if cfg.eval_output_dir is not None
            else None
        )
        if (
            position_path is not None
            and action_path is not None
            and position_path.exists()
            and action_path.exists()
            and not evaluation_overwrite()
        ):
            print(f"[skip] {name}: existing probe metrics", flush=True)
            continue
        training_cfg = OmegaConf.load(config_path)
        training_cfg.data.dataset.num_steps = (
            training_cfg.history_size + training_cfg.num_preds
        )
        loader = make_validation_loader(training_cfg, cfg.data.dataset.name)
        checkpoint_path = config_path.parent / f"weights_epoch_{cfg.eval_epoch}.pt"
        model = load_evaluation_model(
            f"{evaluation_checkpoint_run(name)}/weights_epoch_{cfg.eval_epoch}.pt",
            checkpoint_path,
            cache_dir,
        ).to(device).eval()
        data = extract_probe_data(
            model, loader, device, num_batches=cfg.probe_num_batches
        )
        probe_kwargs = dict(
            train_fraction=cfg.probe_train_fraction,
            alpha=cfg.probe_alpha,
            seed=cfg.probe_seed,
        )
        tags = {
            "run": name,
            "arm": arm.name,
            "seed": seed,
            "label_frac": arm.label_frac,
            "cycle_ratio": arm.cycle_ratio,
            "epoch": cfg.eval_epoch,
        }

        metrics = position_probe_metrics(
            data["z_t"], data["state"], data["state_delta"], **probe_kwargs
        )
        show(name, "position", metrics)
        result = {**tags, **metrics}
        results.append(result)

        action_metrics = action_probe_metrics(data, **probe_kwargs)
        show(name, "action", action_metrics)
        action_result = {**tags, **action_metrics}
        action_results.append(action_result)
        if position_path is not None and action_path is not None:
            position_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([result]).to_csv(position_path, index=False)
            pd.DataFrame([action_result]).to_csv(action_path, index=False)

    if cfg.eval_output_dir is not None:
        output_dir = Path(cfg.eval_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if results:
            suffix = "" if selected_runs is None else "_" + "__".join(sorted(selected_runs))
            pd.DataFrame(results).to_csv(
                output_dir / f"position_probe_metrics{suffix}.csv", index=False
            )
            pd.DataFrame(action_results).to_csv(
                output_dir / f"action_probe_metrics{suffix}.csv", index=False
            )


if __name__ == "__main__":
   run()
