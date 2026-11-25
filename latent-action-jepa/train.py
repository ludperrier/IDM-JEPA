import os
import signal
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F  # new
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from module import make_episode_action_mask, make_grouped_action_mask
from utils import (
    get_action_label_mask,
    get_column_normalizer,
    get_img_preprocessor,
    index_key_set,
    keep_episode_fraction,
    keep_aligned_clips,
    remap_legacy_vit_on_load,
    transition_index,
    SaveCkptCallback,
)


def resolve_resume(cfg, run_dir):
    last_ckpt = run_dir / "last.ckpt"
    policy = cfg.get("resume_policy", "auto")
    restarts = int(os.environ.get("SLURM_RESTART_COUNT") or 0)

    if policy not in ("auto", "always", "never"):
        raise ValueError(f"resume_policy must be auto|always|never, got {policy!r}")

    if policy == "never":
        return None, "never"
    if not last_ckpt.exists():
        # nothing local to continue; on a requeue the W&B run may already exist
        return None, ("allow" if restarts else "never")
    if policy == "always" or restarts:
        return last_ckpt, "allow"

    raise FileExistsError(
        f"{last_ckpt} already exists, but this is not a SLURM requeue "
        f"(SLURM_RESTART_COUNT={restarts}). Continuing would splice this training "
        f"onto an earlier one, in both the checkpoint and the W&B run.\n"
        f"  start a clean sweep      : export SWEEP_TAG=v2   (namespaces checkpoints and W&B ids)\n"
        f"  genuinely continue this  : resume_policy=always\n"
        f"  discard and retrain      : rm -rf {run_dir}\n"
    )


class PersistentLastCheckpoint(pl.Callback):
    def __init__(self, path, every_n_train_steps=1000):
        self.path = path
        self.every_n_train_steps = every_n_train_steps
        self.last_saved_step = None

    def save(self, trainer):
        trainer.save_checkpoint(self.path, weights_only=False)
        self.last_saved_step = trainer.global_step

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero and self.last_saved_step is None:
            self.save(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            trainer.is_global_zero
            and trainer.global_step
            and trainer.global_step % self.every_n_train_steps == 0
            and trainer.global_step != self.last_saved_step
        ):
            self.save(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero and trainer.global_step != self.last_saved_step:
            self.save(trainer)


def cycle_loss(model, actions, ctx_emb, next_emb, label_mask, cfg):
    assert cfg.space == "action", f"only the action-space variant is implemented, got {cfg.space}"  # new
    labeled_actions = actions[label_mask]
    if not len(labeled_actions):
        return None

    all_ctx = ctx_emb.flatten(0, 1)
    all_next = next_emb.flatten(0, 1)
    action_idx = torch.randint(
        len(labeled_actions), (len(all_ctx),), device=ctx_emb.device
    )

    a = labeled_actions[action_idx].detach().unsqueeze(1)
    z_t = all_ctx.unsqueeze(1)
    real_delta = all_next.unsqueeze(1) - z_t

    a_emb = model.action_encoder(a)  # new
    z_hat = model.predict(z_t, a_emb)  # new  imagined next state

    # z_hat is never detached: the predictor is the module the cycle loss repairs
    code_emb, aux = model.infer_code_pair(z_t, z_hat, detach=False)  # new
    if model.shares_action_encoder:  # new
        pred_a, tgt_a = aux["code"], a  # new  raw action space
    else:  # new
        pred_a, tgt_a = code_emb, a_emb  # new  no shared raw space; fall back to embeddings
    loss = (pred_a.float() - tgt_a.float().detach()).pow(2).mean()  # new

    return {
        "loss": loss,
        "z_t": z_t,
        "z_hat": z_hat,
        "a_emb": a_emb,
        "real_delta": real_delta,
    }


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight
    cyc_cfg = cfg.loss.get("cycle", None)  # new
    action_group_size = int(cfg.data.dataset.frameskip)
    action_dim = batch["action"].size(-1)
    if action_dim % action_group_size:
        raise ValueError(
            f"action width {action_dim} is not divisible by frameskip "
            f"{action_group_size}"
        )
    raw_action_dim = action_dim // action_group_size
    raw_actions = batch["action"].unflatten(-1, (action_group_size, raw_action_dim))
    raw_action_valid = ~torch.isnan(raw_actions[:, :ctx_len]).any(dim=-1)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]
    ctx_act = act_emb[:, :ctx_len]

    if self.model.idm is not None:
        code_emb, code_aux = self.model.infer_code(emb)
        episode_idx, step_idx = transition_index(batch, cfg, ctx_len)
        if cfg.action_label_mode == "grouped":
            raw_label_mask = make_grouped_action_mask(
                episode_idx,
                step_idx,
                action_group_size,
                cfg.label_frac,
                label_seed=cfg.label_seed,
                valid=raw_action_valid,
            )
        elif cfg.action_label_mode == "episode":
            raw_label_mask = make_episode_action_mask(
                episode_idx,
                action_group_size,
                cfg.label_frac,
                label_seed=cfg.label_seed,
                valid=raw_action_valid,
            )
        else:
            raise ValueError(
                "action_label_mode must be grouped|episode, got "
                f"{cfg.action_label_mode!r}"
            )
        if self.model.shares_action_encoder:
            code, act = code_aux["code"][:, :ctx_len], batch["action"][:, :ctx_len]
            component_mask = raw_label_mask.repeat_interleave(raw_action_dim, dim=-1)
            mixed_action = torch.where(component_mask, act, code)
            ctx_act = self.model.action_encoder(mixed_action)
        else:
            raise ValueError(
                "grouped raw-action masking requires IDM codes and actions to "
                "share the action encoder"
            )
        per_component = (code.float() - act.float().detach()).pow(2)
        output["action_teacher_loss"] = (
            per_component * component_mask
        ).sum() / component_mask.sum().clamp(min=1)
        label_rate = raw_label_mask.float().mean()
        fully_labeled_block = raw_label_mask.all(dim=-1)

    ctx_emb = emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred
    cycle_scale = 0.0
    if (
        cyc_cfg is not None
        and self.model.idm is not None
        and cyc_cfg.ratio > 0
    ):
        epochs_since_warmup = self.current_epoch - cyc_cfg.warmup_epochs
        ramp_epochs = cyc_cfg.get("ramp_epochs", 0)
        if ramp_epochs:
            cycle_scale = min(max(epochs_since_warmup / ramp_epochs, 0.0), 1.0)
        elif epochs_since_warmup >= 0:
            cycle_scale = 1.0
    cyc = None
    if cycle_scale > 0:
        cyc = cycle_loss(
            self.model,
            batch["action"][:, :ctx_len],
            ctx_emb,
            emb[:, n_preds : n_preds + ctx_len],
            fully_labeled_block,
            cyc_cfg,
        )
        if cyc is not None:
            output["cycle_loss"] = cyc["loss"]

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))

    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    if "action_teacher_loss" in output:
        output["loss"] = output["loss"] + cfg.loss.action_teacher.weight * output["action_teacher_loss"]
    if "cycle_loss" in output:  # new
        w_cyc = cycle_scale * cyc_cfg.ratio * cfg.loss.action_teacher.weight
        output["loss"] = output["loss"] + w_cyc * output["cycle_loss"]  # modified

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    if self.model.idm is not None:
        losses_dict[f"{stage}/label_rate"] = label_rate.detach()
        losses_dict[f"{stage}/fully_labeled_block_rate"] = (
            fully_labeled_block.float().mean().detach()
        )
    if cyc_cfg is not None and self.model.idm is not None and cyc_cfg.ratio > 0:
        losses_dict[f"{stage}/cycle_scale"] = torch.tensor(
            cycle_scale, device=emb.device
        )
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    if cyc is not None:  # new
        with torch.no_grad():  # new
            real_d = cyc["real_delta"].float()  # new
            imag_d = (cyc["z_hat"] - cyc["z_t"]).float()  # new
            # falsifier for the "labeled data is the anti-collapse mechanism" bet:  # new
            # if these two delta distributions separate, the anchor has stopped  # new
            # binding and any cycle-loss gain after that point is suspect  # new
            metrics = {  # new
                "cycle/delta_norm_ratio": imag_d.norm(dim=-1).mean()  # new
                / real_d.norm(dim=-1).mean().clamp(min=1e-6),  # new
                "cycle/delta_cos": F.cosine_similarity(imag_d, real_d, dim=-1).mean(),  # new
                # same z_t, different action -> free action-sensitivity probe.  # new
                # rising sensitivity with flat planning success = watermarking  # new
                "cycle/action_sensitivity": (
                    cyc["z_hat"] - pred_emb.flatten(0, 1).unsqueeze(1)
                ).float().norm(dim=-1).mean()
                / pred_emb.float().norm(dim=-1).mean().clamp(min=1e-6),  # new
                "cycle/act_emb_norm": cyc["a_emb"].float().norm(dim=-1).mean(),  # new
            }  # new
        self.log_dict({f"{stage}/{k}": v for k, v in metrics.items()}, on_step=True, sync_dist=True)  # new

    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################


    index_keys = index_key_set(cfg)
    with open_dict(cfg):
        for key in index_keys:
            if key not in cfg.data.dataset.keys_to_load:
                cfg.data.dataset.keys_to_load.append(key)
            if key not in cfg.data.dataset.keys_to_cache:
                cfg.data.dataset.keys_to_cache.append(key)

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    try:
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
    except KeyError as err:
        raise KeyError(
            f"{err}. The label mask needs the columns named by episode_key="
            f"{cfg.episode_key!r} and step_key={cfg.step_key!r}. Some PushT "
            f"exports spell the first one 'ep_idx' -- override episode_key if so."
        ) from err
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    episode_row_mask = None
    if cfg.episode_frac < 1.0:
        episode_row_mask, kept_episodes, total_episodes, before, after = keep_episode_fraction(
            dataset, cfg.episode_key, cfg.episode_frac, cfg.episode_seed
        )
        print(
            f"[episodes] keep {kept_episodes} / {total_episodes} episodes; "
            f"{after} / {before} clips (seed={cfg.episode_seed})",
            flush=True,
        )
    action_label_mask = get_action_label_mask(
        dataset,
        cfg.episode_key,
        cfg.step_key,
        cfg.label_frac,
        cfg.label_seed,
        action_block_size=cfg.data.dataset.frameskip,
        label_mode=cfg.action_label_mode,
    )
    print(
        f"[labels] action normalization uses {action_label_mask.sum().item()} / "
        f"{action_label_mask.numel()} globally labeled raw actions "
        f"(mode={cfg.action_label_mode})",
        flush=True,
    )

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            # index columns are identifiers, not features: z-scoring them would
            # destroy the transition identity the label mask is keyed on
            if col.startswith("pixels") or col in index_keys:
                continue
            normalizer = get_column_normalizer(
                dataset,
                col,
                col,
                row_mask=(
                    action_label_mask if episode_row_mask is None else action_label_mask & episode_row_mask
                ) if col == "action" else episode_row_mask,
                empty_identity=col == "action",
            )
            transforms.append(normalizer)
        # one "action" fed to the model is a frameskip-long block of raw actions
        action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
        cfg.model.action_encoder.input_dim = action_dim
        cfg.code_dim = action_dim

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    if cfg.non_overlapping_actions:
        before, after = keep_aligned_clips(dataset)
        print(
            f"[labels] non-overlapping clips keep {after} / {before} starts "
            f"(alignment={cfg.data.dataset.frameskip})",
            flush=True,
        )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    warmup_steps = 1
    if cfg.trainer.max_epochs <= warmup_steps:
        raise ValueError(
            f"trainer.max_epochs={cfg.trainer.max_epochs} must exceed the scheduler "
            f"warmup of {warmup_steps} epoch(s), otherwise LinearWarmupCosineAnnealingLR "
            f"divides by zero. Use max_epochs>={warmup_steps + 1}."
        )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": warmup_steps,
                "max_steps": cfg.trainer.max_epochs,
            },
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    world_model.register_load_state_dict_pre_hook(remap_legacy_vit_on_load)

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    resume_ckpt, wandb_resume = resolve_resume(cfg, run_dir)
    print(
        f"[resume] policy={cfg.get('resume_policy', 'auto')} "
        f"restart_count={os.environ.get('SLURM_RESTART_COUNT') or 0} "
        f"ckpt={resume_ckpt} wandb_resume={wandb_resume}",
        flush=True,
    )

    logger = None
    if cfg.wandb.enabled:
        wandb_settings = dict(cfg.wandb.config)
        # "never" makes W&B refuse an id that already exists, so a duplicate
        # submission fails loudly instead of quietly extending the old run
        wandb_settings["resume"] = wandb_resume
        logger = WandbLogger(**wandb_settings)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    resume_ckpt_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename=f"{cfg.output_model_name}_weights",
        save_last=True,
        every_n_epochs=1,
    )

    plugins = None
    if os.environ.get("SLURM_JOB_ID") is not None:
        plugins = [SLURMEnvironment(auto_requeue=True, requeue_signal=signal.SIGTERM)]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            object_dump_callback,
            resume_ckpt_callback,
            PersistentLastCheckpoint(
                run_dir / "last.ckpt",
                every_n_train_steps=cfg.get("checkpoint_interval_steps", 1000),
            ),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
        plugins=plugins,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=resume_ckpt,
    )

    manager()
    return


if __name__ == "__main__":
    run()