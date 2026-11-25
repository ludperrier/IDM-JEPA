#BASE LEWM Implementation (LeWorldModel, Mael et al, 2026)
import json
import os
os.environ["MUJOCO_GL"] = "egl"
import time
from pathlib import Path


import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from utils import get_action_label_mask, remap_legacy_vit_state_dict


def load_policy_model(name):
    try:
        return swm.wm.utils.load_pretrained(name)
    except RuntimeError:
        checkpoint_path = Path(
            swm.data.utils.get_cache_dir(sub_folder="checkpoints"), name
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not any(key.startswith("encoder.encoder.layer.") for key in state_dict):
            raise

        with (checkpoint_path.parent / "config.json").open() as config_file:
            model = hydra.utils.instantiate(json.load(config_file))

        model.load_state_dict(remap_legacy_vit_state_dict(state_dict))
        return model


def img_transform(cfg):
   transform = transforms.Compose(
       [
           transforms.ToImage(),
           transforms.ToDtype(torch.float32, scale=True),
           transforms.Normalize(**spt.data.dataset_stats.ImageNet),
           transforms.Resize(size=cfg.eval.img_size),
       ]
   )
   return transform

def get_episodes_length(dataset, episodes):
   col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


   episode_idx = dataset.get_col_data(col_name)
   step_idx = dataset.get_col_data("step_idx")
   lengths = []
   for ep_id in episodes:
       lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
   return np.array(lengths)

def get_dataset(cfg, dataset_name):
   dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
   dataset = swm.data.HDF5Dataset(
       dataset_name,
       keys_to_cache=cfg.dataset.keys_to_cache,
       cache_dir=dataset_path,
   )
   return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
   """Run evaluation of dinowm vs random policy."""
   assert (
       cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
   ), "Planning horizon must be smaller than or equal to eval_budget"


   # create world environment
   cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
   world = swm.World(**cfg.world, image_shape=(224, 224))


   # create the transform
   transform = {
       "pixels": img_transform(cfg),
       "goal": img_transform(cfg),
   }


   dataset = get_dataset(cfg, cfg.eval.dataset_name)
   stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
   col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
   ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)
   action_label_mask = get_action_label_mask(
       stats_dataset,
       cfg.episode_key,
       cfg.step_key,
       cfg.label_frac,
       cfg.label_seed,
       action_block_size=(
           cfg.plan_config.action_block if cfg.non_overlapping_actions else 1
       ),
   ).cpu().numpy()


   process = {}
   for col in cfg.dataset.keys_to_cache:
       if col in ["pixels"]:
           continue
       processor = preprocessing.StandardScaler()
       col_data = stats_dataset.get_col_data(col)
       if col == "action":
           col_data = col_data[action_label_mask]
       col_data = col_data[~np.isnan(col_data).any(axis=1)]
       if len(col_data):
           processor.fit(col_data)
       elif col == "action":
           # No action labels means action normalization must not inspect hidden
           # labels; fitting zeros gives StandardScaler's identity transform.
           processor.fit(np.zeros((1, stats_dataset.get_dim("action"))))
       else:
           raise ValueError(f"{col}: no finite values available for normalization")
       process[col] = processor


       if col != "action":
           process[f"goal_{col}"] = process[col]


   # -- run evaluation
   policy = cfg.get("policy", "random")


   if policy != "random":
       model = load_policy_model(cfg.policy)
       model = model.to("cuda")
       model = model.eval()
       model.requires_grad_(False)
       model.interpolate_pos_encoding = True
       config = swm.PlanConfig(**cfg.plan_config)
       solver = hydra.utils.instantiate(cfg.solver, model=model)
       policy = swm.policy.WorldModelPolicy(
           solver=solver, config=config, process=process, transform=transform
       )


   else:
       policy = swm.policy.RandomPolicy()


   results_path = (
       Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
       if cfg.policy != "random"
       else Path(__file__).parent
   )


   # sample the episodes and the starting indices
   episode_len = get_episodes_length(dataset, ep_indices)
   max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
   max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
   # Map each dataset row’s episode_idx to its max_start_idx
   col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
   max_start_per_row = np.array(
       [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
   )


   # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
   valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
   valid_indices = np.nonzero(valid_mask)[0]
   print(valid_mask.sum(), "valid starting points found for evaluation.")

   if len(valid_indices) < cfg.eval.num_eval:
       raise ValueError(
           f"Requested {cfg.eval.num_eval} evaluation episodes, but only "
           f"{len(valid_indices)} valid starting points are available."
       )

   g = np.random.default_rng(cfg.seed)
   random_episode_indices = g.choice(valid_indices, size=cfg.eval.num_eval, replace=False)


   # sort increasingly to avoid issues with HDF5Dataset indexing
   random_episode_indices = np.sort(random_episode_indices)


   print(random_episode_indices)


   eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
   eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]


   world.set_policy(policy)


   results_path.mkdir(parents=True, exist_ok=True)


   start_time = time.time()
   metrics = world.evaluate(
       dataset=dataset,
       start_steps=eval_start_idx.tolist(),
       goal_offset=cfg.eval.goal_offset_steps,
       eval_budget=cfg.eval.eval_budget,
       episodes_idx=eval_episodes.tolist(),
       callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
       video=results_path,
   )
   end_time = time.time()
  
   print(metrics)


   results_path = results_path / cfg.output.filename
   results_path.parent.mkdir(parents=True, exist_ok=True)


   with results_path.open("w") as f:
       f.write("==== CONFIG ====\n")
       f.write(OmegaConf.to_yaml(cfg))
       f.write("\n")


       f.write("==== RESULTS ====\n")
       f.write(f"metrics: {metrics}\n")
       f.write(f"evaluation_time: {end_time - start_time} seconds\n")

if __name__ == "__main__":
   run()

