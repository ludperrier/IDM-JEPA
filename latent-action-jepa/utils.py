import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from module import make_mask, transition_hash


LEGACY_VIT_KEY_REPLACEMENTS = {
    ".attention.attention.query.": ".attention.q_proj.",
    ".attention.attention.key.": ".attention.k_proj.",
    ".attention.attention.value.": ".attention.v_proj.",
    ".attention.output.dense.": ".attention.o_proj.",
    ".intermediate.dense.": ".mlp.fc1.",
    ".output.dense.": ".mlp.fc2.",
}


def remap_legacy_vit_state_dict(state_dict):
    """Translate Transformers 4.x ViT parameter names to the 5.x layout."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key.replace("encoder.encoder.layer.", "encoder.layers.")
        for old, new in LEGACY_VIT_KEY_REPLACEMENTS.items():
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def remap_legacy_vit_on_load(
    module, state_dict, prefix, local_metadata, strict, missing_keys,
    unexpected_keys, error_messages,
):
    if not any("encoder.encoder.layer." in key for key in state_dict):
        return
    state_dict.update(remap_legacy_vit_state_dict(state_dict))
    for key in [key for key in state_dict if "encoder.encoder.layer." in key]:
        del state_dict[key]


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str, row_mask=None, empty_identity=False):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    if row_mask is not None:
        row_mask = torch.as_tensor(row_mask, dtype=torch.bool)
        if row_mask.numel() != data.size(0):
            raise ValueError(
                f"{source}: label mask has {row_mask.numel()} rows, expected {data.size(0)}"
            )
        data = data[row_mask]
    data = data[~torch.isnan(data).any(dim=1)]
    if not len(data):
        if not empty_identity:
            raise ValueError(f"{source}: no finite values available for normalization")
        width = torch.from_numpy(np.array(col_data)).shape[-1]
        mean = torch.zeros(1, width)
        std = torch.ones(1, width)
        return dt.transforms.WrapTorchTransform(
            ZScoreNormalizer(mean, std), source=source, target=target
        )
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


def get_action_label_mask(
    dataset,
    episode_key,
    step_key,
    label_frac,
    label_seed,
    action_block_size=1,
    label_mode="grouped",
):
    """Return the dataset-row action-label budget used by every action-dependent path."""
    actions = torch.from_numpy(np.array(dataset.get_col_data("action")))
    valid = ~torch.isnan(actions).any(dim=1)
    episode_idx = torch.from_numpy(np.array(dataset.get_col_data(episode_key)))
    step_idx = torch.from_numpy(np.array(dataset.get_col_data(step_key)))
    if label_mode == "episode":
        step_idx = torch.zeros_like(step_idx)
    elif label_mode == "grouped" and action_block_size > 1:
        step_idx = step_idx - torch.remainder(step_idx, action_block_size)
    elif label_mode != "grouped":
        raise ValueError(
            f"action_label_mode must be grouped|episode, got {label_mode!r}"
        )
    return make_mask(episode_idx, step_idx, label_frac, label_seed=label_seed, valid=valid)


def keep_episode_fraction(dataset, episode_key, fraction, seed):
    """Keep an exact, seed-stable fraction of complete episodes in a dataset."""
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"episode_frac must be in (0, 1], got {fraction}")

    episode_idx = torch.from_numpy(np.array(dataset.get_col_data(episode_key))).to(torch.int64)
    episode_ids = torch.unique(episode_idx, sorted=True)
    num_keep = int(fraction * len(episode_ids))
    if not num_keep:
        raise ValueError(
            f"episode_frac={fraction} selects no episodes from {len(episode_ids)} available"
        )

    ranks = transition_hash(episode_ids, torch.zeros_like(episode_ids), label_seed=seed)
    selected_ids = episode_ids[torch.argsort(ranks)[:num_keep]]
    row_mask = torch.isin(episode_idx, selected_ids)
    selected = {int(episode_id) for episode_id in selected_ids.tolist()}
    before = len(dataset.clip_indices)
    dataset.clip_indices = [
        clip for clip in dataset.clip_indices if int(clip[0]) in selected
    ]
    if not dataset.clip_indices:
        raise ValueError("episode selection removed every dataset clip")
    return row_mask, len(selected_ids), len(episode_ids), before, len(dataset.clip_indices)


def keep_aligned_clips(dataset):
    """Keep clips whose action blocks start on non-overlapping frameskip boundaries."""
    frameskip = int(dataset.frameskip)
    if frameskip < 1:
        raise ValueError(f"frameskip must be positive, got {frameskip}")
    before = len(dataset.clip_indices)
    dataset.clip_indices = [
        clip for clip in dataset.clip_indices if clip[1] % frameskip == 0
    ]
    if not dataset.clip_indices:
        raise ValueError("non-overlapping clip selection removed every dataset clip")
    return before, len(dataset.clip_indices)

def index_key_set(cfg):
    return {cfg.get("episode_key", "episode_idx"), cfg.get("step_key", "step_idx")}


def transition_index(batch, cfg, ctx_len):
    resolved = []
    for key in (cfg.episode_key, cfg.step_key):
        if key not in batch:
            raise KeyError(
                f"column {key!r} missing from the batch; the deterministic label "
                f"mask needs it. Check keys_to_load and the episode_key/step_key "
                f"overrides (some PushT exports use 'ep_idx')."
            )
        value = batch[key]
        if value.dim() == 3 and value.size(-1) == 1:
            value = value.squeeze(-1)
        if value.dim() != 2:
            raise ValueError(
                f"expected {key!r} to be (B, T) or (B, T, 1), got {tuple(batch[key].shape)}"
            )
        resolved.append(value[:, :ctx_len].to(torch.int64))
    return resolved


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )