"""Current decomposition utilities for trained mRNN checkpoints."""

from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from modeling.utils.models import Model


DEFAULT_CONDITION_COLUMNS = [
    "high_interactivity_face",
    "low_interactivity_face",
    "object",
]
CONDITION_LABELS = {
    "high_interactivity_face": "interactive face",
    "low_interactivity_face": "noninteractive face",
    "object": "object",
}


def load_model_from_checkpoint(model_dir, *, device="cpu"):
    """Rebuild a trained Model from a checkpoint directory."""
    model_dir = Path(model_dir)
    with (model_dir / "hp.json").open("r", encoding="utf-8") as f:
        hp = json.load(f)

    checkpoint_path = model_dir / f"{hp['model_save_name']}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    region_unit_counts = checkpoint.get("region_unit_counts") or hp.get(
        "region_unit_counts"
    )
    if region_unit_counts is None:
        raise ValueError(
            "Checkpoint must include region_unit_counts, or pass through a loader "
            "that can recover them from the original dataset."
        )

    model = Model(
        hp["mrnn_config_file"],
        hp.get("hidden_size", 100),
        _region_count(region_unit_counts, "dmpfc", "pfc"),
        _region_count(region_unit_counts, "accg", "acc"),
        region_unit_counts["ofc"],
        region_unit_counts["bla"],
        hp["dt"],
        hp["tau"],
        hp.get("inp_noise", 0.0),
        hp.get("act_noise", 0.0),
        hp.get("rec_constrained", False),
        hp.get("inp_constrained", False),
        hp.get("batch_first", True),
        hp.get("spectral_radius"),
        output_layer=hp.get("output_layer", True),
        latent_training=hp.get("latent_training", False),
        n_components=hp.get("n_components", 10),
        device=str(device),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, hp, checkpoint


def _region_count(region_unit_counts, region, legacy_region=None):
    """Fetch a region count, accepting one legacy alias for older checkpoints."""
    if region in region_unit_counts:
        return region_unit_counts[region]
    if legacy_region is not None and legacy_region in region_unit_counts:
        return region_unit_counts[legacy_region]
    raise KeyError(f"Missing region count for {region}")


def build_condition_input(condition_names, timesteps, *, device):
    """Create the 3-channel condition input used by the mRNN training code."""
    input_tensor = torch.zeros(
        (len(condition_names), timesteps, 3), dtype=torch.float32, device=device
    )
    condition_index = {
        "high_interactivity_face": 0,
        "low_interactivity_face": 1,
        "object": 2,
    }
    for row_idx, condition_name in enumerate(condition_names):
        input_tensor[row_idx, :, condition_index[condition_name]] = 1.0
    return input_tensor


def condition_label(condition):
    """Return a readable condition label for plotting."""
    return CONDITION_LABELS.get(condition, condition)


def recurrent_weight(model):
    """Return the effective recurrent weight matrix used in the forward pass."""
    mrnn = model.mrnn
    w_rec, w_rec_mask, w_rec_sign = mrnn.gen_w(mrnn.region_dict)
    if mrnn.constrained:
        w_rec = mrnn.apply_dales_law(
            w_rec,
            w_rec_mask,
            w_rec_sign,
            lower_bound=mrnn.lower_bound_rec,
            upper_bound=mrnn.upper_bound_rec,
        )
    return w_rec


def input_weight(model):
    """Return the effective input weight matrix used in the forward pass."""
    mrnn = model.mrnn
    w_inp, w_inp_mask, w_inp_sign = mrnn.gen_w(mrnn.inp_dict)
    if mrnn.constrained:
        w_inp = mrnn.apply_dales_law(
            w_inp,
            w_inp_mask,
            w_inp_sign,
            lower_bound=mrnn.lower_bound_inp,
            upper_bound=mrnn.upper_bound_inp,
        )
    return w_inp


def region_slice(model, region):
    """Return a slice into the full hidden vector for one recurrent region."""
    start_idx = 0
    for cur_region, region_data in model.mrnn.region_dict.items():
        end_idx = start_idx + region_data.num_units
        if cur_region == region:
            return slice(start_idx, end_idx)
        start_idx = end_idx
    raise KeyError(f"Unknown recurrent region: {region}")


def replay_checkpoint(model_dir, *, timesteps, condition_columns=None, device="cpu"):
    """Run a saved model without noise and return state/activity trajectories."""
    model, hp, checkpoint = load_model_from_checkpoint(model_dir, device=device)
    condition_columns = condition_columns or checkpoint.get(
        "condition_columns", DEFAULT_CONDITION_COLUMNS
    )
    inp = build_condition_input(condition_columns, timesteps, device=device)
    xn_0 = checkpoint.get("xn_0")
    if xn_0 is None:
        xn_0 = torch.zeros((1, model.mrnn.total_num_units), device=device)
    else:
        xn_0 = xn_0.to(device)

    with torch.no_grad():
        x_seq, h_seq = model.mrnn(xn_0, inp, noise=False)
        out, _ = model(xn_0, inp, noise=False)

    return {
        "model": model,
        "hp": hp,
        "checkpoint": checkpoint,
        "condition_columns": condition_columns,
        "inp": inp,
        "xn_0": xn_0,
        "x_seq": x_seq,
        "h_seq": h_seq,
        "out": out,
    }


def decompose_currents(replay):
    """Compute source-region current vectors into each target region over time."""
    model = replay["model"]
    mrnn = model.mrnn
    x_seq = replay["x_seq"]
    h_seq = replay["h_seq"]
    inp = replay["inp"]
    xn_0 = replay["xn_0"].expand(inp.shape[0], -1)
    h_0 = mrnn.activation(xn_0)

    x_prev = torch.cat([xn_0.unsqueeze(1), x_seq[:, :-1]], dim=1)
    h_prev = torch.cat([h_0.unsqueeze(1), h_seq[:, :-1]], dim=1)

    w_rec = recurrent_weight(model)
    w_inp = input_weight(model)
    baseline = mrnn.get_tonic_inp()
    regions = list(mrnn.region_dict)

    rows = []
    vectors = {}
    for target_region in regions:
        target_slice = region_slice(model, target_region)
        target_next = h_seq[..., target_slice]
        leak = -x_prev[..., target_slice]
        base = baseline[target_slice].expand_as(target_next)
        external = (w_inp[target_slice, :] @ inp.reshape(-1, inp.shape[-1]).T).T
        external = external.reshape(inp.shape[0], inp.shape[1], -1)

        vectors[(target_region, "leak")] = leak.detach().cpu()
        vectors[(target_region, "baseline")] = base.detach().cpu()
        vectors[(target_region, "external_input")] = external.detach().cpu()

        for source_region in regions:
            source_slice = region_slice(model, source_region)
            block = w_rec[target_slice, source_slice]
            source_h = h_prev[..., source_slice]
            current = (block @ source_h.reshape(-1, source_h.shape[-1]).T).T
            current = current.reshape(source_h.shape[0], source_h.shape[1], -1)
            vectors[(target_region, source_region)] = current.detach().cpu()

            rows.extend(
                _current_summary_rows(
                    current,
                    target_next,
                    replay["condition_columns"],
                    target_region,
                    source_region,
                )
            )

        rows.extend(
            _current_summary_rows(
                external,
                target_next,
                replay["condition_columns"],
                target_region,
                "external_input",
            )
        )
        rows.extend(
            _current_summary_rows(
                leak,
                target_next,
                replay["condition_columns"],
                target_region,
                "leak",
            )
        )
        rows.extend(
            _current_summary_rows(
                base,
                target_next,
                replay["condition_columns"],
                target_region,
                "baseline",
            )
        )

    return pd.DataFrame(rows), vectors


def pairwise_current_alignment(replay, vectors, *, sources=None):
    """Compare source-current vectors pairwise within each target region."""
    model = replay["model"]
    regions = list(model.mrnn.region_dict)
    sources = sources or regions
    rows = []
    for target_region in regions:
        for i, source_a in enumerate(sources):
            for source_b in sources[i + 1 :]:
                vec_a = vectors[(target_region, source_a)]
                vec_b = vectors[(target_region, source_b)]
                cos = F.cosine_similarity(vec_a, vec_b, dim=-1, eps=1e-8)
                for cond_idx, condition in enumerate(replay["condition_columns"]):
                    for time_idx in range(cos.shape[1]):
                        rows.append(
                            {
                                "condition": condition,
                                "time_idx": time_idx,
                                "target_region": target_region,
                                "source_a": source_a,
                                "source_b": source_b,
                                "cosine": float(cos[cond_idx, time_idx]),
                            }
                        )
    return pd.DataFrame(rows)


def recurrent_current_contribution_table(current_df, regions):
    """Normalize recurrent-current magnitudes across source regions per target/time."""
    recurrent = current_df[current_df["source"].isin(regions)].copy()
    group_cols = ["condition", "time_idx", "target_region"]
    total = recurrent.groupby(group_cols)["current_norm"].transform("sum")
    recurrent["relative_current_norm"] = np.where(
        total > 0, recurrent["current_norm"] / total, 0.0
    )
    return recurrent


def current_alignment_summary(current_df, regions):
    """Average source-current magnitude and alignment to next target activity."""
    return (
        current_df[current_df["source"].isin(regions)]
        .groupby(["condition", "target_region", "source"], as_index=False)
        .agg(
            mean_current_norm=("current_norm", "mean"),
            mean_cosine_to_next_activity=("cosine_to_next_activity", "mean"),
        )
    )


def pairwise_current_alignment_summary(pairwise_df):
    """Average pairwise source-current alignment within target regions."""
    return (
        pairwise_df.groupby(
            ["condition", "target_region", "source_a", "source_b"], as_index=False
        ).agg(mean_pairwise_cosine=("cosine", "mean"))
    )


def plot_relative_recurrent_currents(
    relative_df, regions, condition_names, *, plt_module=None
):
    """Plot relative recurrent-current magnitude by source, target, and condition."""
    if plt_module is None:
        import matplotlib.pyplot as plt_module

    n_rows = len(regions)
    n_cols = len(condition_names)
    fig, axes = plt_module.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 2.5 * n_rows), sharex=True, sharey=True
    )
    axes = np.asarray(axes)
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for row_idx, target_region in enumerate(regions):
        for col_idx, condition in enumerate(condition_names):
            ax = axes[row_idx, col_idx]
            subset = relative_df[
                (relative_df["target_region"] == target_region)
                & (relative_df["condition"] == condition)
            ]
            pivot = (
                subset.pivot_table(
                    index="time_idx",
                    columns="source",
                    values="relative_current_norm",
                    aggfunc="mean",
                )
                .reindex(columns=regions)
                .fillna(0.0)
                .sort_index()
            )
            ax.stackplot(pivot.index.to_numpy(), pivot.T.to_numpy(), labels=pivot.columns)
            if row_idx == 0:
                ax.set_title(condition_label(condition))
            if col_idx == 0:
                ax.set_ylabel(f"target {target_region}\nrelative norm")
            if row_idx == n_rows - 1:
                ax.set_xlabel("time bin")
            ax.set_ylim(0, 1)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
    )
    fig.tight_layout()
    return fig


def _current_summary_rows(current, target_next, conditions, target_region, source):
    norm = torch.linalg.vector_norm(current, dim=-1)
    target_norm = torch.linalg.vector_norm(target_next, dim=-1)
    cosine = F.cosine_similarity(current, target_next, dim=-1, eps=1e-8)
    dot = (current * target_next).sum(dim=-1)
    rows = []
    for cond_idx, condition in enumerate(conditions):
        for time_idx in range(current.shape[1]):
            rows.append(
                {
                    "condition": condition,
                    "time_idx": time_idx,
                    "target_region": target_region,
                    "source": source,
                    "current_norm": float(norm[cond_idx, time_idx]),
                    "target_next_norm": float(target_norm[cond_idx, time_idx]),
                    "dot_to_next_activity": float(dot[cond_idx, time_idx]),
                    "cosine_to_next_activity": float(cosine[cond_idx, time_idx]),
                }
            )
    return rows


def analyze_checkpoint_currents(
    model_dir,
    *,
    timesteps,
    condition_columns=None,
    device="cpu",
    save=True,
):
    """Run current analysis and optionally save CSV outputs next to checkpoint."""
    replay = replay_checkpoint(
        model_dir,
        timesteps=timesteps,
        condition_columns=condition_columns,
        device=device,
    )
    current_df, vectors = decompose_currents(replay)
    pairwise_df = pairwise_current_alignment(replay, vectors)

    if save:
        out_dir = Path(model_dir) / "current_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        current_df.to_csv(out_dir / "source_to_next_activity_alignment.csv", index=False)
        pairwise_df.to_csv(out_dir / "pairwise_source_current_alignment.csv", index=False)

    return replay, current_df, pairwise_df, vectors
