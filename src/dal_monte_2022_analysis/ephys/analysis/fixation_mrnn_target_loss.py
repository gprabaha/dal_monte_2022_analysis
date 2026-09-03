"""Making the fitted mRNN reproduce the data better, and checking it is still one model.

Task 00 settled how to train. This module settles *what to train on*. The selected
protocol reproduces the region PC trajectories well in aggregate but has two specific
failures, both traced to the objective rather than to the architecture:

- **Interactive face is under-fitted.** The loss is an absolute MSE, so a condition's
  influence is proportional to its target energy. Interactive face carries about 11% of
  that energy -- it has five times more trials than the other conditions and therefore
  the cleanest, lowest-variance estimate -- yet it produces roughly half the residual.
- **Fast structure is dropped, and only there.** The model reproduces 84-99% of the
  observed power above 10 Hz for the other conditions and about 15% for interactive
  face. That content is verified signal: split-half reliability of the region PCs runs
  from 0.998 down to 0.90 across all 42 components.

Two knobs address that directly (condition balancing and PC whitening), and two more
change what the dynamics can express (spectral radius, now that it is actually applied,
and hidden width). This module defines the variants, and scores them on three axes at
once -- fit against the noise ceiling, recovery of high-frequency power, and agreement
between independently seeded fits. The third matters because a variant that fits better
while making seeds disagree has not improved the model, it has only made it easier to
overfit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_protocol import (
    resolve_chapter_root,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import (
    backproject_replay_outputs_to_firing_rates,
    extract_fixation_latent_dynamics,
    extract_region_current_vectors,
    reconstruction_accuracy,
    replay_fixation_mrnn_run,
)
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_training import (
    FixationMRNNRunSettings,
    load_fixation_mrnn_config,
    settings_from_config,
)

CONDITION_ORDER: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")

#: Frequency bands the spectral-recovery score is reported over. The 10-20 Hz band is
#: where the selected protocol loses interactive face, so it is the one to watch.
FREQUENCY_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 5.0, "0-5 Hz"),
    (5.0, 10.0, "5-10 Hz"),
    (10.0, 20.0, "10-20 Hz"),
    (20.0, 50.0, "20-50 Hz"),
)


@dataclass(frozen=True)
class TargetLossVariant:
    """One candidate target/loss configuration."""

    label: str
    condition_loss_weighting: str = "uniform"
    pc_loss_weighting: str = "uniform"
    spectral_radius: float = 1.1
    hidden_units: int = 50
    temporal_derivative_loss_scale: float = 1.0
    temporal_curvature_loss_scale: float = 0.5

    def overrides(self) -> dict[str, object]:
        return {
            "condition_loss_weighting": str(self.condition_loss_weighting),
            "pc_loss_weighting": str(self.pc_loss_weighting),
            "spectral_radius": float(self.spectral_radius),
            "hidden_units": int(self.hidden_units),
            "temporal_derivative_loss_scale": float(self.temporal_derivative_loss_scale),
            "temporal_curvature_loss_scale": float(self.temporal_curvature_loss_scale),
        }


def target_loss_variants(
    *,
    condition_modes: Sequence[str] = ("uniform", "balanced"),
    pc_modes: Sequence[str] = ("uniform", "sqrt_whiten", "whiten"),
    dynamics_probe: Sequence[tuple[float, int]] = ((1.4, 50), (1.1, 80), (1.4, 80)),
    probe_condition_mode: str = "balanced",
    probe_pc_mode: str = "sqrt_whiten",
) -> list[TargetLossVariant]:
    """The variant grid: a weighting factorial plus a small dynamics probe.

    The weighting arm is the factorial that tests the diagnosis directly. The dynamics
    probe asks a different question -- whether the model *can* express the missing fast
    structure at all -- by widening the network and raising the spectral radius, which
    lengthens how long its modes persist. It is run at the best-guess weighting rather
    than crossed with everything, because if the weighting arm closes the gap the probe
    is unnecessary, and if it does not, the probe is the next thing to try.

    The first entry is always the task-00 baseline, so every comparison has a reference
    fitted under identical conditions rather than inherited from a different sweep.
    """
    variants: list[TargetLossVariant] = []
    for condition_mode in condition_modes:
        for pc_mode in pc_modes:
            variants.append(
                TargetLossVariant(
                    label=f"cond{condition_mode}_pc{pc_mode}",
                    condition_loss_weighting=condition_mode,
                    pc_loss_weighting=pc_mode,
                )
            )
    for spectral_radius, hidden_units in dynamics_probe:
        variants.append(
            TargetLossVariant(
                label=f"probe_sr{spectral_radius:g}".replace(".", "p") + f"_h{int(hidden_units)}",
                condition_loss_weighting=probe_condition_mode,
                pc_loss_weighting=probe_pc_mode,
                spectral_radius=float(spectral_radius),
                hidden_units=int(hidden_units),
            )
        )
    return variants


def baseline_label(variants: Sequence[TargetLossVariant]) -> str:
    """The task-00 configuration, which is the uniform/uniform cell."""
    for variant in variants:
        if variant.condition_loss_weighting == "uniform" and variant.pc_loss_weighting == "uniform":
            return variant.label
    return variants[0].label


def load_selected_protocol(path: str | Path) -> dict[str, object]:
    """The recipe task 00 froze, which every variant here inherits."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_variant_settings(
    variant: TargetLossVariant,
    *,
    protocol: Mapping[str, object],
    mrnn_cfg_path: str | Path,
    seed: int,
) -> FixationMRNNRunSettings:
    """Settings for one variant: the task-00 recipe with this variant's overrides."""
    base = settings_from_config(load_fixation_mrnn_config(mrnn_cfg_path))
    architecture = dict(protocol["architecture"])
    optimizer = dict(protocol["optimizer"])
    # The variant owns these; the protocol's values are the baseline it overrides.
    for key in ("spectral_radius", "temporal_derivative_loss_scale", "temporal_curvature_loss_scale", "hidden_units"):
        architecture.pop(key, None)
        optimizer.pop(key, None)
    return replace(
        base,
        **architecture,
        **optimizer,
        **variant.overrides(),
        epochs=int(protocol["epochs"]),
        seed=int(seed),
        divergence_loss_threshold=None,
    )


def variant_run_dir(root: str | Path, variant: TargetLossVariant, seed: int) -> Path:
    return Path(root) / variant.label / f"seed={int(seed)}"


def variant_job_commands(
    variants: Sequence[TargetLossVariant],
    seeds: Sequence[int],
    *,
    root: str | Path,
    repo_root: str | Path,
    protocol: Mapping[str, object],
    mrnn_cfg_path: str | Path,
    conda_env: str = "gaze_processing",
    skip_existing: bool = True,
) -> tuple[list[str], list[Path]]:
    """One shell command per missing (variant, seed) cell."""
    repo_root = Path(repo_root)
    commands: list[str] = []
    run_dirs: list[Path] = []
    for variant in variants:
        for seed in seeds:
            run_dir = variant_run_dir(root, variant, seed)
            run_dirs.append(run_dir)
            if skip_existing and (run_dir / "checkpoint_best.pth").exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            settings = build_variant_settings(
                variant, protocol=protocol, mrnn_cfg_path=mrnn_cfg_path, seed=seed
            )
            payload = {key: value for key, value in asdict(settings).items()}
            payload["region_order"] = list(payload["region_order"])
            payload["condition_order"] = list(payload["condition_order"])
            config_path = run_dir / "run_config.yaml"
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=True)
            commands.append(
                " ".join(
                    [
                        f"cd {repo_root} &&",
                        "FIXATION_MRNN_PROGRESS=off",
                        f"conda run -n {conda_env} python",
                        "scripts/ephys/modeling/train_fixation_mrnn_into_run_dir.py",
                        f"--mrnn-cfg {config_path}",
                        f"--run-dir {run_dir}",
                        f"--seed {int(seed)}",
                        "--device auto",
                        "--overwrite",
                    ]
                )
            )
    return commands, run_dirs


def index_variant_runs(
    root: str | Path,
    variants: Sequence[TargetLossVariant],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """One row per (variant, seed) cell with its completion state."""
    rows: list[dict[str, object]] = []
    for variant in variants:
        for seed in seeds:
            run_dir = variant_run_dir(root, variant, seed)
            complete = (run_dir / "checkpoint_best.pth").exists()
            failed = (run_dir / "training_failed.json").exists()
            rows.append(
                {
                    **asdict(variant),
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "complete": bool(complete),
                    "diverged": bool(failed and not complete),
                    "pending": not complete and not failed,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Scoring axis 1: fit, against the ceiling rather than against 1.0
# --------------------------------------------------------------------------------------


def ceiling_relative_fit(
    run_dir: str | Path,
    ceiling_by_region: Mapping[str, float],
    *,
    device: str = "cpu",
) -> pd.DataFrame:
    """Region x condition $R^2$, and the same divided by what the data determines.

    Scoring against 1.0 asks the model to reproduce sampling noise. ``ceiling_by_region``
    is the mean split-half reliability of that region's PC trajectories; a model at 1.0
    on the corrected scale has extracted everything the data supports, and above 1.0 it
    is fitting noise.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    accuracy = reconstruction_accuracy(replay)
    accuracy["ceiling"] = [float(ceiling_by_region.get(str(r), np.nan)) for r in accuracy["region"]]
    accuracy["r2_vs_ceiling"] = accuracy["r2"] / accuracy["ceiling"]
    return accuracy.assign(run_dir=str(run_dir))


# --------------------------------------------------------------------------------------
# Scoring axis 2: does the model reproduce the fast structure?
# --------------------------------------------------------------------------------------


def _band_power(traces: np.ndarray, *, bin_size_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean power spectrum over features for one ``(time, features)`` panel."""
    centred = traces - traces.mean(axis=0, keepdims=True)
    window = np.hanning(centred.shape[0])[:, None]
    spectrum = np.abs(np.fft.rfft(centred * window, axis=0)) ** 2
    return np.fft.rfftfreq(centred.shape[0], d=bin_size_s), spectrum.mean(axis=1)


def spectral_recovery(
    run_dir: str | Path,
    *,
    space: str = "fr",
    device: str = "cpu",
    bands: Sequence[tuple[float, float, str]] = FREQUENCY_BANDS,
) -> pd.DataFrame:
    """Fraction of the observed power the model reproduces, by condition and band.

    A ratio near 1 means the model tracks the observed fluctuations at that timescale; a
    ratio near 0 means it has replaced them with something smoother. Reported in
    PC-backprojected firing-rate space by default, since that is where the deficit was
    first seen by eye.
    """
    replay = replay_fixation_mrnn_run(Path(run_dir), device=device)
    checkpoint = replay["checkpoint"]
    timeline = np.asarray(checkpoint["timeline_s"], dtype=float)
    bin_size_s = float(np.mean(np.diff(timeline)))
    conditions = tuple(replay["condition_order"])
    predicted_fr = backproject_replay_outputs_to_firing_rates(replay) if space == "fr" else None

    rows: list[dict[str, object]] = []
    for region in replay["region_order"]:
        if space == "fr":
            observed = np.asarray(checkpoint["pc_reconstructed_raw_by_region"][region], dtype=float)
            predicted = np.asarray(predicted_fr[region], dtype=float)
        else:
            observed = np.asarray(checkpoint["target_by_region"][region], dtype=float)
            predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float)
        for index, condition in enumerate(conditions):
            freqs, observed_power = _band_power(observed[index], bin_size_s=bin_size_s)
            _, predicted_power = _band_power(predicted[index], bin_size_s=bin_size_s)
            for low, high, name in bands:
                selection = (freqs >= low) & (freqs < high)
                denominator = float(observed_power[selection].sum())
                rows.append(
                    {
                        "region": region,
                        "condition": condition,
                        "band": name,
                        "space": space,
                        "power_ratio": float(predicted_power[selection].sum() / denominator)
                        if denominator > 0
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows).assign(run_dir=str(run_dir))


# --------------------------------------------------------------------------------------
# Scoring axis 3: do independently seeded fits agree?
# --------------------------------------------------------------------------------------


def _flat_outputs(replay: Mapping[str, object]) -> np.ndarray:
    return np.concatenate(
        [replay["output_by_region"][region].detach().cpu().numpy().ravel() for region in replay["region_order"]]
    )


def _flat_current_magnitudes(replay: Mapping[str, object]) -> np.ndarray:
    """Rotation- and permutation-invariant summary of the inter-regional drive."""
    currents = extract_region_current_vectors(replay)
    regions = tuple(replay["region_order"])
    parts = [
        np.linalg.norm(currents[(source, target)].numpy(), axis=-1).ravel()
        for source in regions
        for target in regions
        if (source, target) in currents
    ]
    return np.concatenate(parts)


def _flat_drive_geometry(replay: Mapping[str, object]) -> np.ndarray:
    """Upper triangle of each region's condition x time drive dissimilarity matrix."""
    model = replay["model"]
    slices = model.hidden_region_slices()
    latent = extract_fixation_latent_dynamics(replay)
    parts = []
    for region in replay["region_order"]:
        drive = np.stack(
            [latent[c]["recurrent_drive"][:, slices[region]].numpy() for c in replay["condition_order"]],
            axis=0,
        )
        flat = drive.reshape(-1, drive.shape[-1])
        centred = flat - flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centred, axis=1, keepdims=True)
        normalized = np.divide(centred, np.maximum(norms, 1e-8), out=np.zeros_like(centred), where=norms > 1e-8)
        rdm = 1.0 - np.clip(normalized @ normalized.T, -1.0, 1.0)
        parts.append(rdm[np.triu_indices(rdm.shape[0], k=1)])
    return np.concatenate(parts)


#: The three summaries seed agreement is measured on, coarsest first. Outputs is the
#: weakest test -- two models can produce the same trajectories by different means -- and
#: geometry the strongest.
SEED_AGREEMENT_FEATURES: dict[str, object] = {
    "output trajectories": _flat_outputs,
    "inter-regional current magnitude": _flat_current_magnitudes,
    "latent drive geometry": _flat_drive_geometry,
}


def inter_seed_agreement(
    run_dirs: Sequence[str | Path],
    *,
    device: str = "cpu",
    features: Mapping[str, object] = SEED_AGREEMENT_FEATURES,
) -> pd.DataFrame:
    """Mean pairwise correlation between independently seeded fits of one variant.

    Hidden units are permutable and sign-ambiguous across initializations, so raw weights
    cannot be compared; each summary here is invariant to that relabelling. A variant that
    improves the fit while lowering this has bought accuracy with degrees of freedom, which
    is the trade the whole modelling programme has to avoid.
    """
    replays = [replay_fixation_mrnn_run(Path(run_dir), device=device) for run_dir in run_dirs]
    rows: list[dict[str, object]] = []
    for name, extractor in features.items():
        vectors = [np.asarray(extractor(replay), dtype=float) for replay in replays]
        if len({v.shape for v in vectors}) != 1:
            # Variants differing in hidden width give different-length summaries; only
            # the output trajectories are comparable across them, and within a variant
            # every seed has the same shape, which is the case this is used in.
            rows.append({"feature": name, "n_seeds": len(vectors), "mean_agreement": np.nan,
                         "min_agreement": np.nan, "n_pairs": 0})
            continue
        pairs = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                a, b = vectors[i] - vectors[i].mean(), vectors[j] - vectors[j].mean()
                denominator = np.sqrt(np.sum(a**2) * np.sum(b**2))
                pairs.append(float(np.sum(a * b) / denominator) if denominator > 0 else np.nan)
        rows.append(
            {
                "feature": name,
                "n_seeds": len(vectors),
                "n_pairs": len(pairs),
                "mean_agreement": float(np.nanmean(pairs)) if pairs else np.nan,
                "min_agreement": float(np.nanmin(pairs)) if pairs else np.nan,
            }
        )
    return pd.DataFrame(rows)


def score_variants(
    inventory: pd.DataFrame,
    ceiling_by_region: Mapping[str, float],
    *,
    device: str = "cpu",
    watch_band: str = "10-20 Hz",
    watch_condition: str = "face_interactive",
) -> pd.DataFrame:
    """One row per variant, on all three axes.

    Deliberately not collapsed into a single number: the axes trade against one another,
    and which trade is acceptable is a judgement about what the chapter needs rather than
    something a weighted sum can settle.
    """
    rows: list[dict[str, object]] = []
    for label, block in inventory[inventory["complete"].astype(bool)].groupby("label", sort=False):
        run_dirs = list(block["run_dir"])
        fit = pd.concat([ceiling_relative_fit(d, ceiling_by_region, device=device) for d in run_dirs])
        recovery = pd.concat([spectral_recovery(d, device=device) for d in run_dirs])
        agreement = inter_seed_agreement(run_dirs, device=device).set_index("feature")
        watch = recovery[(recovery["band"] == watch_band) & (recovery["condition"] == watch_condition)]
        interactive = fit[fit["condition"] == "face_interactive"]
        rows.append(
            {
                "label": label,
                "n_seeds": len(run_dirs),
                "mean_r2_vs_ceiling": float(fit["r2_vs_ceiling"].mean()),
                "interactive_r2_vs_ceiling": float(interactive["r2_vs_ceiling"].mean()),
                "worst_cell_r2_vs_ceiling": float(fit["r2_vs_ceiling"].min()),
                f"power_recovered_{watch_condition}_{watch_band.replace(' ', '')}": float(watch["power_ratio"].mean()),
                "seed_agreement_outputs": float(agreement.loc["output trajectories", "mean_agreement"]),
                "seed_agreement_currents": float(agreement.loc["inter-regional current magnitude", "mean_agreement"]),
                "seed_agreement_geometry": float(agreement.loc["latent drive geometry", "mean_agreement"]),
            }
        )
    return pd.DataFrame(rows)


def resolve_task_root(cfg_path: str | Path = "configs/dataset.yaml") -> Path:
    """Output root for this task's runs."""
    return resolve_chapter_root(cfg_path, task="01_target_and_loss")


__all__ = [
    "CONDITION_ORDER",
    "FREQUENCY_BANDS",
    "SEED_AGREEMENT_FEATURES",
    "TargetLossVariant",
    "baseline_label",
    "build_variant_settings",
    "ceiling_relative_fit",
    "index_variant_runs",
    "inter_seed_agreement",
    "load_selected_protocol",
    "resolve_task_root",
    "score_variants",
    "spectral_recovery",
    "target_loss_variants",
    "variant_job_commands",
    "variant_run_dir",
]
