import os
os.environ.setdefault("MUJOCO_GL", "egl")

import json
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import numpy as np
import pandas as pd
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn.manifold import TSNE
from stable_worldmodel.wm.utils import load_pretrained
from torch.utils.data import DataLoader

from evaluate_next_state import (
    available_evaluation_runs,
    evaluation_checkpoint_run,
    evaluation_run_filter,
    evaluation_seeds,
)
from utils import get_img_preprocessor, remap_legacy_vit_state_dict


QUANTITIES = {
    "agent_x": (0, "Agent x-position", "viridis"),
    "agent_y": (1, "Agent y-position", "viridis"),
    "block_x": (2, "Block x-position", "viridis"),
    "block_y": (3, "Block y-position", "viridis"),
    "block_orientation": (4, "Block orientation", "twilight"),
}

ORIENTATION_INDEX = 4


def load_geometry_model(name, checkpoint_path, cache_dir):
    try:
        return load_pretrained(name, cache_dir=cache_dir)
    except RuntimeError:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not any(key.startswith("encoder.encoder.layer.") for key in state_dict):
            raise
        with (checkpoint_path.parent / "config.json").open() as config_file:
            model = hydra.utils.instantiate(json.load(config_file))
        model.load_state_dict(remap_legacy_vit_state_dict(state_dict))
        return model


def make_geometry_loader(cfg, dataset_name):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
        **dataset_cfg,
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", cfg.img_size)
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    _, validation_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    return DataLoader(
        validation_set, **cfg.loader, shuffle=False, drop_last=False
    )


@torch.inference_mode()
def extract_embeddings(model, loader, device, max_samples):
    embeddings = []
    states = []
    collected = 0
    for batch in loader:
        pixels = batch["pixels"].float().to(device)
        encoded = model.encode({"pixels": pixels})["emb"][:, 0]
        state = batch["state"][:, 0].float()
        if state.size(-1) < 5:
            raise ValueError(
                "Expected PushT state=[agent_x, agent_y, block_x, block_y, "
                f"block_orientation], got shape {tuple(state.shape)}."
            )
        valid = ~torch.isnan(state).any(dim=-1)
        encoded = encoded[valid.to(device)].cpu()
        state = state[valid]
        remaining = max_samples - collected
        embeddings.append(encoded[:remaining])
        states.append(state[:remaining])
        collected += min(len(encoded), remaining)
        if collected >= max_samples:
            break
    if collected < 2:
        raise ValueError(f"Need at least two valid held-out samples, got {collected}.")
    return torch.cat(embeddings), torch.cat(states)


def pca_geometry(embeddings):
    centered = embeddings.double() - embeddings.double().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    coordinates = centered @ eigenvectors[:, :2]
    return eigenvalues.float(), explained.float(), coordinates.float()


def tsne_geometry(embeddings, max_samples, seed):
    if max_samples < 2:
        raise ValueError("geometry_tsne_num_samples must be at least 2.")
    generator = torch.Generator().manual_seed(seed)
    sample_count = min(len(embeddings), max_samples)
    indices = torch.randperm(len(embeddings), generator=generator)[:sample_count]
    sampled_embeddings = embeddings[indices].numpy()
    perplexity = min(30.0, float(sample_count - 1))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(sampled_embeddings)
    return indices, torch.from_numpy(coordinates)


def plot_geometry(
    name, quantity_name, color_map, quantity, physical_state, eigenvalues,
    explained, coordinates, output,
):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    components = np.arange(1, len(explained) + 1)
    color_min, color_max = np.quantile(quantity.numpy(), [0.01, 0.99])

    axes[0].semilogy(components, explained.numpy().clip(min=1e-12), color="#245b45")
    axes[0].set(xlabel="Principal component", ylabel="Explained variance ratio", title="PCA spectrum")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(
        physical_state[:, 2].numpy(),
        physical_state[:, 3].numpy(),
        c=quantity.numpy(),
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        s=5,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axes[1].set(
        xlabel="Block x-position",
        ylabel="Block y-position",
        title=f"Physical geometry colored by {quantity_name.lower()}",
    )
    axes[1].set_aspect("equal", adjustable="box")

    scatter = axes[2].scatter(
        coordinates[:, 0].numpy(),
        coordinates[:, 1].numpy(),
        c=quantity.numpy(),
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        s=5,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axes[2].set(xlabel="PC1", ylabel="PC2", title=f"Latent geometry colored by {quantity_name.lower()}")
    figure.colorbar(scatter, ax=axes[2], label=quantity_name)
    figure.suptitle(name, fontsize=13)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_geometry_tsne(
    name, quantity_name, color_map, quantity, coordinates, output,
):
    figure, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    color_min, color_max = np.quantile(quantity.numpy(), [0.01, 0.99])
    scatter = axis.scatter(
        coordinates[:, 0].numpy(),
        coordinates[:, 1].numpy(),
        c=quantity.numpy(),
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        s=5,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axis.set(
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        title=f"Latent t-SNE colored by {quantity_name.lower()}",
    )
    figure.colorbar(scatter, ax=axis, label=quantity_name)
    figure.suptitle(name, fontsize=13)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_geometry_3d(
    name, quantity_name, color_map, quantity, physical_state, coordinates, output,
):
    orientation = physical_state[:, ORIENTATION_INDEX].numpy()
    figure = plt.figure(figsize=(12, 5.6), constrained_layout=True)
    color_min, color_max = np.quantile(quantity.numpy(), [0.01, 0.99])

    axis_physical = figure.add_subplot(1, 2, 1, projection="3d")
    axis_physical.scatter(
        physical_state[:, 2].numpy(),
        physical_state[:, 3].numpy(),
        orientation,
        c=quantity.numpy(),
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        s=5,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axis_physical.set(
        xlabel="Block x-position",
        ylabel="Block y-position",
        zlabel="Block orientation",
        title=f"Physical geometry (3D) colored by {quantity_name.lower()}",
    )

    axis_latent = figure.add_subplot(1, 2, 2, projection="3d")
    scatter = axis_latent.scatter(
        coordinates[:, 0].numpy(),
        coordinates[:, 1].numpy(),
        orientation,
        c=quantity.numpy(),
        cmap=color_map,
        vmin=color_min,
        vmax=color_max,
        s=5,
        alpha=0.45,
        linewidths=0,
        rasterized=True,
    )
    axis_latent.set(
        xlabel="PC1",
        ylabel="PC2",
        zlabel="Block orientation",
        title=f"Latent geometry (3D) colored by {quantity_name.lower()}",
    )
    figure.colorbar(scatter, ax=axis_latent, label=quantity_name, shrink=0.7)
    figure.suptitle(name, fontsize=13)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_spectrum_comparison(spectra, output):
    figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    for name, explained in spectra.items():
        components = np.arange(1, len(explained) + 1)
        axis.semilogy(
            components,
            explained.numpy().clip(min=1e-12),
            label=name,
            linewidth=1.4,
            alpha=0.85,
        )
    axis.set(
        xlabel="Principal component",
        ylabel="Explained variance ratio",
        title="Held-out latent PCA spectra",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7, ncol=2, frameon=False)
    figure.savefig(output, dpi=220)
    plt.close(figure)


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    if cfg.geometry_quantity not in QUANTITIES:
        raise ValueError(
            f"geometry_quantity must be one of {sorted(QUANTITIES)}, "
            f"got {cfg.geometry_quantity!r}."
        )
    if cfg.eval_output_dir is None:
        raise ValueError("eval_output_dir must be set.")
    if cfg.geometry_num_samples < 2:
        raise ValueError("geometry_num_samples must be at least 2.")

    cache_dir = os.environ["STABLEWM_HOME"]
    output_root = Path(cfg.eval_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantity_index, quantity_name, color_map = QUANTITIES[cfg.geometry_quantity]
    selected_runs = evaluation_run_filter()
    available = list(
        available_evaluation_runs(
            cache_dir,
            cfg.eval_epoch,
            evaluation_seeds(),
            selected_runs=selected_runs,
        )
    )
    if not available:
        raise FileNotFoundError(
            f"No checkpoints found for SWEEP_TAG={os.environ.get('SWEEP_TAG')!r} "
            f"at epoch {cfg.eval_epoch}."
        )

    summaries = []
    spectra = {}
    for arm, seed, name, config_path in available:
        run_dir = output_root / name
        if (run_dir / "pca_spectrum.csv").exists() and os.environ.get(
            "EVAL_OVERWRITE", "0"
        ).lower() not in {"1", "true", "yes"}:
            print(f"[skip] {name}: existing geometry at {run_dir}", flush=True)
            continue
        training_cfg = OmegaConf.load(config_path)
        training_cfg.data.dataset.num_steps = training_cfg.history_size + training_cfg.num_preds
        loader = make_geometry_loader(training_cfg, cfg.data.dataset.name)
        checkpoint_path = config_path.parent / f"weights_epoch_{cfg.eval_epoch}.pt"
        model = load_geometry_model(
            f"{evaluation_checkpoint_run(name)}/weights_epoch_{cfg.eval_epoch}.pt",
            checkpoint_path,
            cache_dir,
        ).to(device).eval()
        embeddings, states = extract_embeddings(
            model, loader, device, int(cfg.geometry_num_samples)
        )
        eigenvalues, explained, coordinates = pca_geometry(embeddings)
        tsne_indices, tsne_coordinates = tsne_geometry(
            embeddings,
            int(cfg.geometry_tsne_num_samples),
            int(cfg.geometry_tsne_seed),
        )
        spectra[name] = explained
        quantity = states[:, quantity_index]

        run_dir.mkdir(parents=True, exist_ok=True)
        plot_geometry(
            name,
            quantity_name,
            color_map,
            quantity,
            states,
            eigenvalues,
            explained,
            coordinates,
            run_dir / "latent_geometry.png",
        )
        plot_geometry_3d(
            name,
            quantity_name,
            color_map,
            quantity,
            states,
            coordinates,
            run_dir / "latent_geometry_3d.png",
        )
        plot_geometry_tsne(
            name,
            quantity_name,
            color_map,
            quantity[tsne_indices],
            tsne_coordinates,
            run_dir / "latent_geometry_tsne.png",
        )
        pd.DataFrame(
            {
                "component": np.arange(1, len(explained) + 1),
                "eigenvalue": eigenvalues.numpy(),
                "explained_variance_ratio": explained.numpy(),
                "cumulative_explained_variance": explained.cumsum(0).numpy(),
            }
        ).to_csv(run_dir / "pca_spectrum.csv", index=False)
        np.savez_compressed(
            run_dir / "pc_coordinates.npz",
            pc1=coordinates[:, 0].numpy(),
            pc2=coordinates[:, 1].numpy(),
            quantity=quantity.numpy(),
            quantity_name=cfg.geometry_quantity,
        )
        np.savez_compressed(
            run_dir / "tsne_coordinates.npz",
            tsne1=tsne_coordinates[:, 0].numpy(),
            tsne2=tsne_coordinates[:, 1].numpy(),
            quantity=quantity[tsne_indices].numpy(),
            sample_indices=tsne_indices.numpy(),
            quantity_name=cfg.geometry_quantity,
        )
        effective_rank = torch.exp(
            -(explained.double() * explained.double().clamp_min(1e-12).log()).sum()
        ).item()
        summaries.append(
            {
                "run": name,
                "arm": arm.name,
                "seed": seed,
                "label_frac": arm.label_frac,
                "cycle_ratio": arm.cycle_ratio,
                "epoch": cfg.eval_epoch,
                "num_samples": len(embeddings),
                "quantity": cfg.geometry_quantity,
                "pc1_explained": explained[0].item(),
                "pc2_explained": explained[1].item(),
                "pc1_pc2_explained": explained[:2].sum().item(),
                "effective_rank": effective_rank,
                "quantity_mean": quantity.mean().item(),
                "quantity_std": quantity.std(unbiased=False).item(),
            }
        )
        print(
            f"[geometry] {name} samples={len(embeddings)} "
            f"pc1={explained[0]:.6f} pc1_pc2={explained[:2].sum():.6f} "
            f"effective_rank={effective_rank:.3f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(summaries).to_csv(output_root / "latent_geometry_summary.csv", index=False)
    plot_spectrum_comparison(spectra, output_root / "pca_spectra_comparison.png")


if __name__ == "__main__":
    run()