import json
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from stable_worldmodel.wm.utils import load_pretrained
from sweep import ARMS, RAMP_ARMS, SEEDS, RUNS, run_name  # noqa: F401  (RUNS re-exported)
from utils import (
	get_action_label_mask,
	get_column_normalizer,
	get_img_preprocessor,
	index_key_set,
	keep_aligned_clips,
	remap_legacy_vit_state_dict,
)


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


def evaluation_run_filter():
	configured = os.environ.get("EVAL_RUNS")
	if configured is None:
		return None
	runs = {name.strip() for name in configured.split(",") if name.strip()}
	if not runs:
		raise ValueError("EVAL_RUNS must contain at least one run name.")
	return runs


def evaluation_checkpoint_aliases():
	configured = os.environ.get("EVAL_RUN_ALIASES", "")
	aliases = {}
	for entry in configured.split(","):
		if not entry.strip():
			continue
		canonical, separator, checkpoint_run = entry.partition("=")
		if not separator or not canonical or not checkpoint_run:
			raise ValueError(
				"EVAL_RUN_ALIASES entries must have canonical_run=checkpoint_run format."
			)
		aliases[canonical.strip()] = checkpoint_run.strip()
	return aliases


def evaluation_checkpoint_run(run):
	return evaluation_checkpoint_aliases().get(run, run)


def evaluation_overwrite():
	return os.environ.get("EVAL_OVERWRITE", "0").lower() in {"1", "true", "yes"}


def per_run_metrics_path(output_dir, metric_name, run_name):
	return Path(output_dir) / f"{metric_name}_{run_name}.csv"


def make_validation_loader(cfg, dataset_name):
	dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
	dataset_cfg.pop("name")
	dataset = swm.data.load_dataset(
		dataset_name,
		transform=None,
		cache_dir=os.environ.get("LOCAL_DATASET_DIR"),
		**dataset_cfg,
	)
	index_keys = index_key_set(cfg)
	action_block_size = (
		cfg.data.dataset.frameskip if cfg.get("non_overlapping_actions", False) else 1
	)
	action_label_mask = get_action_label_mask(
		dataset,
		cfg.episode_key,
		cfg.step_key,
		cfg.label_frac,
		cfg.label_seed,
		action_block_size=action_block_size,
	)
	transforms = [get_img_preprocessor("pixels", "pixels", cfg.img_size)]
	transforms.extend(
		get_column_normalizer(
			dataset,
			key,
			key,
			row_mask=action_label_mask if key == "action" else None,
			empty_identity=key == "action",
		)
		for key in cfg.data.dataset.keys_to_load
		if not key.startswith("pixels") and key not in index_keys
	)
	dataset.transform = spt.data.transforms.Compose(*transforms)
	if cfg.get("non_overlapping_actions", False):
		keep_aligned_clips(dataset)

	generator = torch.Generator().manual_seed(cfg.seed)
	_, validation_set = spt.data.random_split(
		dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator
	)
	return DataLoader(validation_set, **cfg.loader, shuffle=False, drop_last=False)


def available_evaluation_runs(cache_dir, epoch, seeds, selected_runs=None):
	checkpoint_root = Path(cache_dir) / "checkpoints"
	aliases = evaluation_checkpoint_aliases()
	available = {}
	for seed in seeds:
		arms = (*ARMS, *(RAMP_ARMS if seed == 0 else ()))
		for arm in arms:
			name = run_name(arm, seed)
			if selected_runs is not None and name not in selected_runs:
				continue
			available[name] = (arm, seed)

	tag = os.environ.get("SWEEP_TAG", "").strip()
	if tag:
		for run_dir in checkpoint_root.glob(f"{tag}_*"):
			name = run_dir.name
			if selected_runs is not None and name not in selected_runs:
				continue
			if name in available or not run_dir.is_dir():
				continue
			config = run_dir / "config.yaml"
			if not config.exists():
				continue
			training_cfg = OmegaConf.load(config)
			arm_name = name.removeprefix(f"{tag}_").rsplit("_s", 1)[0]
			seed_text = name.rsplit("_s", 1)[-1]
			try:
				seed = int(seed_text)
			except ValueError:
				continue
			available[name] = (
				SimpleNamespace(
					name=arm_name,
					label_frac=float(training_cfg.get("label_frac", 1.0)),
					cycle_ratio=float(
						OmegaConf.select(training_cfg, "loss.action_teacher.cycle.ratio", default=0.0)
					),
				),
				seed,
			)

	for name, checkpoint_run in aliases.items():
		if selected_runs is not None and name not in selected_runs:
			continue
		run_dir = checkpoint_root / checkpoint_run
		config = run_dir / "config.yaml"
		if not config.exists():
			continue
		training_cfg = OmegaConf.load(config)
		arm_name = name.split("_", 1)[1].rsplit("_s", 1)[0]
		seed_text = name.rsplit("_s", 1)[-1]
		try:
			seed = int(seed_text)
		except ValueError:
			continue
		available[name] = (
			SimpleNamespace(
				name=arm_name,
				label_frac=float(training_cfg.get("label_frac", 1.0)),
				cycle_ratio=float(
					OmegaConf.select(training_cfg, "loss.action_teacher.cycle.ratio", default=0.0)
				),
			),
			seed,
		)

	for name, (arm, seed) in sorted(available.items()):
		run_dir = checkpoint_root / aliases.get(name, name)
		checkpoint = run_dir / f"weights_epoch_{epoch}.pt"
		config = run_dir / "config.yaml"
		if checkpoint.exists() and config.exists():
			yield arm, seed, name, config
		elif selected_runs is not None:
			print(
				f"[skip] {name}: missing "
				f"{'checkpoint' if not checkpoint.exists() else 'config'} for epoch {epoch}",
				flush=True,
			)


def load_evaluation_model(name, checkpoint_path, cache_dir):
	state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
	if not any(key.startswith("encoder.encoder.layer.") for key in state_dict):
		return load_pretrained(name, cache_dir=cache_dir)
	with (checkpoint_path.parent / "config.json").open() as config_file:
		model = hydra.utils.instantiate(json.load(config_file))
	model.load_state_dict(remap_legacy_vit_state_dict(state_dict))
	return model


@torch.inference_mode()
def next_state_metrics(model, loader, device):
	totals = {
		"mse": 0.0,
		"persistence_mse": 0.0,
		"zero_mse": 0.0,
		"action_shuffled_mse": 0.0,
		"state_shuffled_mse": 0.0,
	}
	if model.idm is not None:
		totals["idm_oracle_code_mse"] = 0.0
		totals["idm_shuffled_next_state_code_mse"] = 0.0
	total_elements = 0

	for batch in loader:
		batch = {key: value.to(device) for key, value in batch.items()}
		batch["action"] = torch.nan_to_num(batch["action"], 0.0)
		output = model.encode(batch)
		embeddings = output["emb"]
		action_embeddings = output["act_emb"]

		state_context = embeddings[:, :-1]
		action_context = action_embeddings[:, :-1]
		target = embeddings[:, 1:]
		permutation = torch.roll(torch.arange(len(batch["pixels"]), device=device), 1)

		predictions = {
			"mse": model.predict(state_context, action_context),
			"persistence_mse": state_context,
			"zero_mse": torch.zeros_like(target),
			"action_shuffled_mse": model.predict(state_context, action_context[permutation]),
			"state_shuffled_mse": model.predict(state_context[permutation], action_context),
		}
		if model.idm is not None:
			code, _ = model.infer_code(embeddings)
			shuffled_next_state = torch.cat(
				[embeddings[:, :1], embeddings[permutation, 1:]], dim=1
			)
			shuffled_code, _ = model.infer_code(shuffled_next_state)
			predictions["idm_oracle_code_mse"] = model.predict(state_context, code)
			predictions["idm_shuffled_next_state_code_mse"] = model.predict(
				state_context, shuffled_code
			)

		for name, prediction in predictions.items():
			assert prediction.shape == target.shape, (prediction.shape, target.shape)
			totals[name] += (prediction - target).square().sum().item()
		total_elements += target.numel()

	return {name: total / total_elements for name, total in totals.items()}


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
	cache_dir = os.environ["STABLEWM_HOME"]
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	results = []
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
		result_path = (
			per_run_metrics_path(cfg.eval_output_dir, "next_state_metrics", name)
			if cfg.eval_output_dir is not None
			else None
		)
		if result_path is not None and result_path.exists() and not evaluation_overwrite():
			print(f"[skip] {name}: existing metrics at {result_path}", flush=True)
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
		metrics = next_state_metrics(model, loader, device)
		metrics["persistence_improvement"] = (
			1 - metrics["mse"] / metrics["persistence_mse"]
		)
		print(name, " ".join(f"{k}={v:.6f}" for k, v in metrics.items()), flush=True)
		result = {
			"run": name,
			"arm": arm.name,
			"seed": seed,
			"label_frac": arm.label_frac,
			"cycle_ratio": arm.cycle_ratio,
			"epoch": cfg.eval_epoch,
			**metrics,
		}
		results.append(result)
		if result_path is not None:
			result_path.parent.mkdir(parents=True, exist_ok=True)
			import pandas as pd

			pd.DataFrame([result]).to_csv(result_path, index=False)

	if cfg.eval_output_dir is not None:
		output_dir = Path(cfg.eval_output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)
		import pandas as pd

		if results:
			suffix = "" if selected_runs is None else "_" + "__".join(sorted(selected_runs))
			pd.DataFrame(results).to_csv(
				output_dir / f"next_state_metrics{suffix}.csv", index=False
			)


if __name__ == "__main__":
	run()