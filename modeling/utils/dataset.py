import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np

import sys
from pathlib import Path

# Add the root directory of the repository to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from utils.exp_utils import pca_batched


def normalization(x):
    """Normalize values using robust percentiles with a small stabilizer."""
    # Compute a robust scale using 95th and 5th percentiles to reduce outlier impact.
    scale = np.percentile(x, 95) - np.percentile(x, 5)
    # Add a small stabilizer to avoid division by very small values.
    return x / (scale + 5)


class MeanFixationDataset:
    """Utility wrapper for grouping fixation-locked firing rates by region/condition."""

    def __init__(self, dataframe, batch_first=True, group_by_columns=None):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe containing firing rate data.
            group_by_columns (list): Columns used to group data into batches.
        """
        # Copy to avoid mutating caller-owned data.
        self.dataframe = dataframe.copy()
        # Track whether tensors are organized as (batch, time, features) or (time, batch, features).
        self.batch_first = batch_first
        # This dataset assumes three condition columns by default.
        self.num_conds = 3

        # Use caller-provided condition columns or fall back to defaults.
        self.group_by_columns = group_by_columns or [
            "high_interactivity_face",
            "low_interactivity_face",
            "object",
        ]

        # Ensure missing condition entries won't break grouping/stacking.
        self.dataframe[self.group_by_columns] = self.dataframe[
            self.group_by_columns
        ].fillna("UNKNOWN")
        # Pre-group by region for consistent ordering and per-region bookkeeping.
        df_by_region = self.dataframe.groupby(["region"], sort=False)

        # Build per-region counts and a flattened list of unit IDs in region order.
        self.units_per_region = {}
        self.unit_ids = []
        self.regions = []
        for region, data in df_by_region:
            self.units_per_region[region[0]] = len([unit for unit in data["uuid"]])
            self.unit_ids.extend([unit for unit in data["uuid"]])
            self.regions.append(region)

        # Cache total number of units for convenience.
        self.total_num_units = len(self.unit_ids)

    def __len__(self):
        return len(self.group_by_columns)

    def get_region_indices(self, region):
        """Return the [start, end) indices for a region in the concatenated unit list."""
        # Walk regions in insertion order to compute the slice for the requested region.
        start_idx = 0
        end_idx = 0
        for i, r in enumerate(self.units_per_region):
            if i > 0:
                # Accumulate start index from the previous region's count.
                start_idx += self.units_per_region[r]
            # Extend end index by the current region's unit count.
            end_idx += self.units_per_region[r]
            if r == region:
                break
        # Return a slice so callers can use it directly for tensor indexing.
        return slice(start_idx, end_idx)

    def sample_batch(self, latent_training=False, n_components=10):
        """Build a padded batch of firing rates (or PCA latents) with loss masks."""
        # Group data once per call to preserve region ordering and slicing.
        df_by_region = self.dataframe.groupby(["region"], sort=False)
        normalized_condition_list = []

        # Build a list of normalized firing-rate tensors per condition.
        for cond in self.group_by_columns:
            normalized_condition_list.append(
                normalization(self.get_fr_tensor(df_by_region, cond))
            )

        # Pad across conditions so time dimensions align for batching.
        firing_rate_batch = pad_sequence(
            normalized_condition_list, batch_first=self.batch_first
        )

        if latent_training:
            # Allocate a latent tensor sized by conditions, time, and per-region components.
            if self.batch_first:
                latent_shape = (
                    self.num_conds,
                    firing_rate_batch.shape[1],
                    n_components * len(self.regions),
                )
            else:
                latent_shape = (
                    firing_rate_batch.shape[0],
                    self.num_conds,
                    n_components * len(self.regions),
                )

            latent_data = torch.empty(latent_shape)

            # Replace each region's raw firing rates with PCA latents.
            for i, r in enumerate(self.regions):
                latent_slice = slice(n_components * i, n_components * (i + 1))
                r_slice = self.get_region_indices(r)
                fr_region_tmp = firing_rate_batch[..., r_slice]
                latent_r = pca_batched(
                    fr_region_tmp,
                    batch_first=self.batch_first,
                    n_components=n_components,
                )
                latent_data[..., latent_slice] = torch.from_numpy(latent_r)

            # Use the latent representation as the returned batch.
            firing_rate_batch = latent_data.clone()

        loss_mask_list = []
        # Build loss masks per condition; all ones because padding encodes lengths.
        for normalized_fr in normalized_condition_list:
            loss_mask_shape = (
                normalized_fr.shape[0],
                n_components * len(self.regions),
            )
            loss_mask_list.append(torch.ones(loss_mask_shape))

        # Pad loss masks to match the padded batch layout.
        loss_mask_batch = pad_sequence(
            loss_mask_list,
            batch_first=self.batch_first,
        )

        # Return batch and masks for downstream loss computation.
        return firing_rate_batch, loss_mask_batch

    def get_fr_tensor(self, df, column):
        """Concatenate firing-rate tensors across regions for a given column."""
        fr_dict = {}
        # Convert each group's list of firing-rate vectors into a tensor per region.
        for key, group_data in df:
            fr_dict[key[0]] = torch.stack(
                [torch.tensor(fr) for fr in group_data[column]], dim=-1
            )
        fr_tensor = []
        # Preserve region order by iterating through the dict in insertion order.
        for item in fr_dict:
            fr_tensor.append(fr_dict[item])
        # Concatenate units across regions along the feature dimension.
        fr_tensor = torch.cat(fr_tensor, dim=-1)
        return fr_tensor
