"""Tables the mRNN chapter reads: every number in its prose, built from the final series.

The chapter notebook (``notebooks/mrnn_chapter/04_chapter.ipynb``) does not fit anything
and does not recompute the per-task analyses. It reads the tables tasks 01-03 cached under
``final/<task>/tables/`` and the ten-seed dense and constrained ensembles, and assembles
from them the handful of summaries the chapter's figures need:

* the **ladder** collapsed to region x rung, condition x rung, and the worst cell per rung;
* one region's **reconstructed PC traces** at every rung, averaged over every fit that
  contained the region at that rung -- the picture of a region needing the network;
* the **cost of each constraint** to each fixation type against the dense network, on one
  footing (regions pooled per seed, drop in ceiling-relative R^2), with a bootstrap over
  seeds for the extra cost to interactive face;
* the rebuild's **connectivity-removal arms** rescored per condition on the matched
  per-cell ceiling, so the strict "remove the connections" comparison sits beside the
  rank-constrained one;
* the **pair-lesion** summary: each pair's share of damage, pooled over fixation types.

Everything here is a re-arrangement of stored numbers; the analyses themselves live in
``fixation_mrnn_audit``, ``fixation_mrnn_sweep`` and ``fixation_mrnn_ensemble``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import stats as _stats
from scipy.optimize import linear_sum_assignment

from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_audit import _lesion_groups, annotate_rank_grid, condition_bias, seed_run_dirs
from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_ensemble import _cluster_points, find_fixed_points, seed_of
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import replay_fixation_mrnn_run_with_ablations
from dal_monte_2022_analysis.ephys.analysis.fixation_mrnn_sweep import resolve_task_root, score_variant_fit
from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import replay_fixation_mrnn_run

CONDITION_ORDER: tuple[str, ...] = ("face_interactive", "face_non_interactive", "object")
RUNG_LABELS: dict[int, str] = {0: "alone", 1: "+1 region", 2: "+2 regions", 3: "all four"}

#: The constraints the summary figure compares, in display order. Each names where its fits
#: come from: a ladder rung, a rank-grid cell, or the ensemble arm.
CONSTRAINT_ORDER: tuple[str, ...] = (
    "no inter-regional connections",
    "within-region rank 1",
    "inter-regional rank 10",
    "inter-regional rank 1",
    "selected: within 1, inter-regional 10",
)

__all__: list[str] = ["CONDITION_ORDER", "CONSTRAINT_ORDER", "RUNG_LABELS"]


# ======================================================================================
# 1. Locating and loading the final series
# ======================================================================================


def final_roots(dataset_cfg_path: str | Path) -> dict[str, Path]:
    """Task roots of the final series, plus the chapter's own output root."""
    roots = {task: resolve_task_root(task, dataset_cfg_path, tree="final")
             for task in ("01_ladder", "02_rank_grid", "02b_bottleneck_properties", "03_ensemble", "04_chapter")}
    roots["series"] = roots["01_ladder"].parent
    roots["rebuild_connectivity"] = resolve_task_root("03_connectivity", dataset_cfg_path, tree="chapter")
    return roots


def load_ceiling_by_cell(series_root: str | Path) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """The matched per-cell ceiling task 01 computed, as the two lookups ``score_variant_fit`` takes."""
    cell = pd.read_csv(Path(series_root) / "pc_space_ceiling_by_cell.csv")
    by_region = cell[cell["condition"] == "all"].set_index("region")["reliability"].to_dict()
    by_cell = {(r, c): float(v) for r, c, v in
               cell[cell["condition"] != "all"][["region", "condition", "reliability"]].itertuples(index=False)}
    return by_region, by_cell


def load_final_tables(roots: Mapping[str, Path], *, hidden_units: int = 40) -> dict[str, pd.DataFrame]:
    """Every cached table the chapter reads, keyed by name. Missing files are simply absent."""
    tables: dict[str, pd.DataFrame] = {}
    ladder_tables = Path(roots["01_ladder"]) / "tables"
    grid_tables = Path(roots["02_rank_grid"]) / "tables"
    ensemble_tables = Path(roots["03_ensemble"]) / "tables"
    sources = {
        "ladder_fit": ladder_tables / "ladder_fit_matched.csv",
        "band_recovery": ladder_tables / "band_recovery.csv",
        "grid_fit": grid_tables / "grid_fit.csv",
        "ensemble_fit": ensemble_tables / "ensemble_fit.csv",
        "fit_cost_by_condition": ensemble_tables / "fit_cost_by_condition.csv",
        "lesion_battery": ensemble_tables / "lesion_battery.csv",
        "lesion_ranking_agreement": ensemble_tables / "lesion_ranking_agreement.csv",
        "lesion_ranking_by_condition": ensemble_tables / "lesion_ranking_by_condition.csv",
        "lesion_dynamics": ensemble_tables / "lesion_dynamics.csv",
        "fixed_points_summary": ensemble_tables / "fixed_points_summary.csv",
        "fixed_points_points": ensemble_tables / "fixed_points_points.csv",
        "jacobian_along_trajectory": ensemble_tables / "jacobian_along_trajectory.csv",
        "region_state_properties": ensemble_tables / "region_state_properties.csv",
        "ordering_consistency": ensemble_tables / "ordering_consistency.csv",
        "similarity_summary": ensemble_tables / "similarity_summary.csv",
        "agreement_battery": ensemble_tables / "agreement_battery.csv",
        "condition_similarity": ensemble_tables / "condition_similarity.csv",
    }
    for name, path in sources.items():
        if path.exists() and path.stat().st_size > 1:
            frame = pd.read_csv(path)
            if "seed" in frame.columns:
                frame["seed"] = frame["seed"].astype(str)
            tables[name] = frame
    if "grid_fit" in tables:
        tables["grid_fit"] = annotate_rank_grid(tables["grid_fit"], hidden_units=hidden_units)
    if "ladder_fit" in tables:
        tables["dense_fit"] = tables["ladder_fit"][tables["ladder_fit"]["label"] == "full"].copy()
    return tables


__all__ += ["final_roots", "load_ceiling_by_cell", "load_final_tables"]


# ======================================================================================
# 2. The ladder
# ======================================================================================


def ladder_by_region(ladder_fit: pd.DataFrame) -> pd.DataFrame:
    """Mean ceiling-relative R^2 per region x rung, with the gain from alone to the full network."""
    table = ladder_fit.groupby(["region", "n_partners"])["r2_vs_ceiling"].mean().unstack("n_partners")
    table["gain_alone_to_full"] = table[table.columns.max()] - table[0]
    return table


def ladder_by_condition(ladder_fit: pd.DataFrame, *, region: str | None = None) -> pd.DataFrame:
    """Per fit (label, seed) and condition, the ceiling-relative R^2 -- one region or all regions pooled."""
    block = ladder_fit if region is None else ladder_fit[ladder_fit["region"] == region]
    return (block.groupby(["n_partners", "label", "seed", "condition"])["r2_vs_ceiling"].mean()
            .reset_index())


def ladder_worst_cell(ladder_fit: pd.DataFrame, *, bar: float) -> pd.DataFrame:
    """Per rung: the worst region x condition cell of each fit, and how many fits clear the bar."""
    worst = ladder_fit.groupby(["label", "seed"])["r2_vs_ceiling"].min().reset_index()
    worst = worst.merge(ladder_fit.groupby("label")["n_partners"].first().reset_index(), on="label")
    out = worst.groupby("n_partners")["r2_vs_ceiling"].agg(["mean", "min", "max", "count"])
    out["n_clearing_bar"] = worst.groupby("n_partners")["r2_vs_ceiling"].apply(lambda d: int((d >= bar).sum()))
    return out.reset_index()


def band_residual_by_rung(band_recovery: pd.DataFrame, ladder_fit: pd.DataFrame, *, region: str | None = None) -> pd.DataFrame:
    """Fraction of target power left in the residual, per band and rung, every fit a row."""
    partners = ladder_fit.groupby(["label", "region"])["n_partners"].first().reset_index()
    block = band_recovery.merge(partners, on=["label", "region"], how="inner")
    if region is not None:
        block = block[block["region"] == region]
    return block


def _region_component_r2(replay: Mapping[str, object], region: str) -> np.ndarray:
    """R^2 per (condition, component) of one region, centred per component over pooled condition x time."""
    target = np.asarray(replay["checkpoint"]["target_by_region"][region], dtype=float)
    predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float)[:, : target.shape[1]]
    centred = target - target.mean(axis=(0, 1), keepdims=True)
    sse = ((predicted - target) ** 2).sum(axis=1)
    sst = (centred ** 2).sum(axis=1)
    return 1.0 - sse / np.clip(sst, 1e-12, None)


def ladder_region_traces(
    ladder_fit: pd.DataFrame,
    *,
    region: str,
    indices: Sequence[int],
    device: str = "cpu",
) -> pd.DataFrame:
    """One region's target and reconstructed PC traces at every rung, averaged over the fits at that rung.

    Every fit that contained ``region`` is replayed; its predicted trajectory for the chosen
    components is recorded, and the rung average (over partner sets and seeds) is what the
    chapter draws, with the across-fit standard deviation as the band. The target is
    identical in every fit -- the target builder keeps a region's PCA and scale fixed across
    subsets -- so the observed trace is stored once per rung and is the same in all of them.
    """
    runs = (ladder_fit[ladder_fit["region"] == region]
            .groupby(["label", "seed"])[["run_dir", "n_partners"]].first().reset_index())
    frames = []
    for run in runs.itertuples():
        replay = replay_fixation_mrnn_run(Path(run.run_dir), device=device)
        timeline = np.asarray(replay["checkpoint"]["timeline_s"], dtype=float)
        target = np.asarray(replay["checkpoint"]["target_by_region"][region], dtype=float)
        predicted = replay["output_by_region"][region].detach().cpu().numpy().astype(float)
        r2 = _region_component_r2(replay, region)
        for c, condition in enumerate(replay["condition_order"]):
            for index in indices:
                frames.append(pd.DataFrame({
                    "label": run.label, "seed": str(run.seed), "n_partners": int(run.n_partners),
                    "region": region, "condition": str(condition), "index": int(index),
                    "time_s": timeline[: target.shape[1]],
                    "observed": target[c, :, index], "predicted": predicted[c, : target.shape[1], index],
                    "r2": float(r2[c, index]),
                }))
    long = pd.concat(frames, ignore_index=True)
    out = (long.groupby(["n_partners", "condition", "index", "time_s"])
           .agg(observed=("observed", "first"), predicted=("predicted", "mean"),
                predicted_sd=("predicted", "std"), n_fits=("predicted", "size"), r2_mean=("r2", "mean"))
           .reset_index())
    out["region"] = region
    return out


def ladder_component_gain(ladder_fit: pd.DataFrame, *, region: str, device: str = "cpu") -> pd.DataFrame:
    """Per component and condition: R^2 alone and in the full network, averaged over seeds, and the gain.

    Used to choose which components the trace figure shows and to state where in the
    spectrum the network acts: the leading components are reproduced by a region alone,
    the gain sits in the mid-rank ones.
    """
    rows = []
    for rung in (0, ladder_fit["n_partners"].max()):
        block = ladder_fit[(ladder_fit["region"] == region) & (ladder_fit["n_partners"] == rung)]
        for run in block.groupby(["label", "seed"])["run_dir"].first().reset_index().itertuples():
            replay = replay_fixation_mrnn_run(Path(run.run_dir), device=device)
            r2 = _region_component_r2(replay, region)
            target = np.asarray(replay["checkpoint"]["target_by_region"][region], dtype=float)
            centred = target - target.mean(axis=(0, 1), keepdims=True)
            share = (centred ** 2).sum(axis=(0, 1))
            share = share / share.sum()
            for c, condition in enumerate(replay["condition_order"]):
                for k in range(r2.shape[1]):
                    rows.append({"n_partners": int(rung), "label": run.label, "seed": str(run.seed),
                                 "condition": str(condition), "index": k, "r2": float(r2[c, k]),
                                 "variance_share": float(share[k])})
    long = pd.DataFrame(rows)
    mean = long.groupby(["n_partners", "condition", "index"])[["r2", "variance_share"]].mean().reset_index()
    wide = mean.pivot(index=["condition", "index"], columns="n_partners", values="r2")
    wide.columns = [f"r2_rung{int(c)}" for c in wide.columns]
    wide["gain"] = wide.iloc[:, -1] - wide.iloc[:, 0]
    wide["variance_share"] = mean.groupby(["condition", "index"])["variance_share"].first()
    return wide.reset_index()


def select_trace_components(gain: pd.DataFrame, *, n_leading: int = 1, n_gain: int = 2, within_top: int = 10) -> list[int]:
    """The components the trace figure shows: the leading ones, plus the largest-gain ones among the first ``within_top``.

    The leading components are there because they carry most of the variance and a region
    reproduces them alone; the gain components are where the network acts. Both are chosen
    on the mean gain over the three fixation types, so no fixation type picks its own.
    """
    pooled = gain.groupby("index")["gain"].mean()
    leading = list(range(n_leading))
    candidates = pooled.loc[[k for k in pooled.index if n_leading <= k < within_top]].sort_values(ascending=False)
    return leading + sorted(int(k) for k in candidates.index[:n_gain])


__all__ += ["band_residual_by_rung", "ladder_by_condition", "ladder_by_region", "ladder_component_gain",
            "ladder_region_traces", "ladder_worst_cell", "select_trace_components"]


# ======================================================================================
# 3. The cost of every constraint, on one footing
# ======================================================================================


def _network_fit_per_seed(fit: pd.DataFrame) -> pd.DataFrame:
    """Regions pooled: mean ceiling-relative R^2 per (seed, condition), wide over condition."""
    return fit.groupby(["seed", "condition"])["r2_vs_ceiling"].mean().unstack("condition")


def _single_rung_as_network(ladder_fit: pd.DataFrame) -> pd.DataFrame:
    """The four single-region fits of one seed read as one network with no inter-regional connections.

    Without cross blocks the regions do not interact, so four independent single-region
    fits at the same seed *are* the "no inter-regional connections" architecture; pooling
    them per seed gives it the same per-seed, regions-pooled score every other arm has.
    """
    block = ladder_fit[ladder_fit["n_partners"] == 0]
    return block.groupby(["seed", "condition"])["r2_vs_ceiling"].mean().unstack("condition")


def constraint_cost_table(
    tables: Mapping[str, pd.DataFrame],
    *,
    n_bootstrap: int = 4000,
    seed: int = 0,
) -> pd.DataFrame:
    """Cost of each constraint to each fixation type against the dense network, with bootstrap CIs.

    One row per (constraint, condition). ``cost`` is the drop in ceiling-relative R^2
    (regions pooled per seed, means over seeds); ``fi_extra_cost`` is interactive face's
    cost minus the mean cost of the other two fixation types, with a CI from resampling
    seeds within each arm (the arms are not paired). ``unexplained_ratio`` is the
    scale-free reading: unexplained variance under the constraint over unexplained
    variance in the dense network.
    """
    generator = np.random.default_rng(int(seed))
    dense = tables["dense_fit"]
    grid = tables["grid_fit"]
    hidden = int(grid["rank_within"].max())

    arms: dict[str, pd.DataFrame] = {
        "no inter-regional connections": tables["ladder_fit"][tables["ladder_fit"]["n_partners"] == 0],
        "within-region rank 1": grid[(grid["rank_within"] == 1) & (grid["rank_cross"] == hidden)],
        "inter-regional rank 10": grid[(grid["rank_within"] == hidden) & (grid["rank_cross"] == 10)],
        "inter-regional rank 1": grid[(grid["rank_within"] == hidden) & (grid["rank_cross"] == 1)],
    }
    if "ensemble_fit" in tables:
        arms["selected: within 1, inter-regional 10"] = tables["ensemble_fit"]

    d_fit = _network_fit_per_seed(dense)
    d_un = 1.0 - dense.groupby(["seed", "condition"])["r2"].mean().unstack("condition")
    rows: list[dict[str, object]] = []
    for name, block in arms.items():
        c_fit = _network_fit_per_seed(block)
        c_un = 1.0 - block.groupby(["seed", "condition"])["r2"].mean().unstack("condition")
        conditions = [c for c in CONDITION_ORDER if c in c_fit.columns]
        draws: dict[str, np.ndarray] = {}
        for condition in conditions:
            cf, df = c_fit[condition].to_numpy(float), d_fit[condition].to_numpy(float)
            cu, du = c_un[condition].to_numpy(float), d_un[condition].to_numpy(float)
            boot = np.empty(n_bootstrap)
            ratio = np.empty(n_bootstrap)
            for k in range(n_bootstrap):
                ic = generator.integers(0, len(cf), len(cf))
                idx = generator.integers(0, len(df), len(df))
                boot[k] = df[idx].mean() - cf[ic].mean()
                ratio[k] = cu[ic].mean() / du[idx].mean()
            draws[condition] = boot
            rows.append({
                "constraint": name, "condition": condition, "n_seeds": int(len(cf)),
                "dense_mean": float(df.mean()), "constrained_mean": float(cf.mean()), "constrained_sd": float(cf.std(ddof=1)),
                "cost": float(df.mean() - cf.mean()),
                "cost_ci_low": float(np.percentile(boot, 2.5)), "cost_ci_high": float(np.percentile(boot, 97.5)),
                "unexplained_ratio": float(cu.mean() / du.mean()),
                "unexplained_ratio_ci_low": float(np.percentile(ratio, 2.5)),
                "unexplained_ratio_ci_high": float(np.percentile(ratio, 97.5)),
            })
        others = [c for c in conditions if c != "face_interactive"]
        if "face_interactive" in draws and others:
            diff = draws["face_interactive"] - np.mean([draws[c] for c in others], axis=0)
            fi_cost = next(r["cost"] for r in rows if r["constraint"] == name and r["condition"] == "face_interactive")
            other_cost = np.mean([r["cost"] for r in rows if r["constraint"] == name and r["condition"] in others])
            for r in rows:
                if r["constraint"] == name:
                    r["fi_extra_cost"] = float(fi_cost - other_cost)
                    r["fi_extra_cost_ci_low"] = float(np.percentile(diff, 2.5))
                    r["fi_extra_cost_ci_high"] = float(np.percentile(diff, 97.5))
                    r["p_fi_costs_more"] = float((diff > 0).mean())
                    r["fi_cost_ratio"] = float(fi_cost / other_cost) if other_cost > 0 else np.nan
    return pd.DataFrame(rows)


def constraint_cost_per_seed(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Every seed's cost per constraint and condition, for drawing the marks behind the means."""
    dense_mean = _network_fit_per_seed(tables["dense_fit"]).mean()
    grid = tables["grid_fit"]
    hidden = int(grid["rank_within"].max())
    arms = {
        "no inter-regional connections": tables["ladder_fit"][tables["ladder_fit"]["n_partners"] == 0],
        "within-region rank 1": grid[(grid["rank_within"] == 1) & (grid["rank_cross"] == hidden)],
        "inter-regional rank 10": grid[(grid["rank_within"] == hidden) & (grid["rank_cross"] == 10)],
        "inter-regional rank 1": grid[(grid["rank_within"] == hidden) & (grid["rank_cross"] == 1)],
    }
    if "ensemble_fit" in tables:
        arms["selected: within 1, inter-regional 10"] = tables["ensemble_fit"]
    frames = []
    for name, block in arms.items():
        per = _network_fit_per_seed(block)
        cost = (dense_mean - per).stack().rename("cost").reset_index()
        cost.columns = ["seed", "condition", "cost"]
        frames.append(cost.assign(constraint=name))
    return pd.concat(frames, ignore_index=True)


__all__ += ["constraint_cost_per_seed", "constraint_cost_table"]


# ======================================================================================
# 4. The rebuild's connectivity-removal arms, rescored on the matched ceiling
# ======================================================================================

#: The rebuild's global-structure arms and what each removes. ``full`` is that tree's own
#: dense baseline: same width and recipe, balanced-sum objective rather than minimax.
REMOVAL_ARMS: dict[str, str] = {
    "full": "dense",
    "within_region_only": "no inter-regional connections",
    "cross_plus_self_diagonal": "within-region recurrence reduced to a self-diagonal",
}


def rescore_removal_arms(
    connectivity_root: str | Path,
    ceiling_by_region: Mapping[str, float],
    ceiling_by_cell: Mapping[tuple[str, str], float],
    *,
    arms: Mapping[str, str] = REMOVAL_ARMS,
    device: str = "cpu",
) -> pd.DataFrame:
    """Per region x condition fit of the rebuild's removal arms on the final series' per-cell ceiling.

    The rebuild fitted these under the balanced-sum objective, so their absolute numbers
    are not comparable with the minimax series; their *cost against their own dense
    baseline* is, and that is what the chapter reports. Returns an empty frame when the
    runs are absent.
    """
    root = Path(connectivity_root)
    records = []
    for label in arms:
        for run_dir in seed_run_dirs(root / label):
            records.append({"label": label, "seed": run_dir.name.replace("seed=", ""), "run_dir": str(run_dir), "complete": True})
    if not records:
        return pd.DataFrame()
    fit = score_variant_fit(pd.DataFrame(records), ceiling_by_region, ceiling_by_cell=ceiling_by_cell, device=device)
    fit["seed"] = fit["seed"].astype(str)
    fit["arm"] = fit["label"].map(arms)
    return fit


def removal_cost_table(removal_fit: pd.DataFrame, *, dense_label: str = "full") -> pd.DataFrame:
    """Cost of each removal arm against the rebuild's own dense fits, per fixation type, every seed."""
    if removal_fit.empty:
        return removal_fit
    per = removal_fit.groupby(["label", "seed", "condition"])["r2_vs_ceiling"].mean().reset_index()
    dense_mean = per[per["label"] == dense_label].groupby("condition")["r2_vs_ceiling"].mean()
    out = per[per["label"] != dense_label].copy()
    out["cost"] = out["condition"].map(dense_mean) - out["r2_vs_ceiling"]
    out["arm"] = out["label"].map(REMOVAL_ARMS)
    return out


__all__ += ["REMOVAL_ARMS", "removal_cost_table", "rescore_removal_arms"]


# ======================================================================================
# 5. Lesions and dynamics, collapsed for the chapter
# ======================================================================================


def pair_lesion_share(lesion_battery: pd.DataFrame) -> pd.DataFrame:
    """Each region pair's share of the summed pair-lesion damage, pooled over fixation types, per (arm, seed)."""
    block = lesion_battery[lesion_battery["lesion_kind"] == "bidirectional"]
    per = block.groupby(["arm", "seed", "lesion"])["damage_sse"].sum().reset_index()
    per["share"] = per["damage_sse"] / per.groupby(["arm", "seed"])["damage_sse"].transform("sum")
    return per.rename(columns={"lesion": "pair"})


def lesion_damage_in_r2(lesion_battery: pd.DataFrame) -> pd.DataFrame:
    """Mean damage per lesion family in shared-R^2 units, beside the matched random-weight control.

    ``damage_shared_r2`` divides the added squared error by the whole target's total sum of
    squares, so it reads as "fraction of the target's variance lost". The random control
    silences the same number of weights chosen anywhere in the network.
    """
    block = lesion_battery.copy()
    block["family"] = block["lesion_kind"]
    per_lesion = block.groupby(["arm", "seed", "family", "lesion", "live_weights_removed"])["damage_shared_r2"].sum().reset_index()
    summary = per_lesion.groupby(["arm", "family"])["damage_shared_r2"].agg(["mean", "std", "min", "max"]).reset_index()
    control = per_lesion[per_lesion["family"] == "random control"]
    control_by_size = control.groupby(["arm", "live_weights_removed"])["damage_shared_r2"].mean()
    sizes = per_lesion[per_lesion["family"] != "random control"].groupby(["arm", "family"])["live_weights_removed"].first()
    summary["matched_random_control"] = [
        float(control_by_size.get((arm, int(sizes.get((arm, fam), -1))), np.nan)) if fam != "random control" else np.nan
        for arm, fam in zip(summary["arm"], summary["family"])
    ]
    return summary


def lesion_extent_change(lesion_dynamics: pd.DataFrame) -> pd.DataFrame:
    """State extent after a lesion relative to the intact trajectory, per (arm, seed, lesion, condition)."""
    block = lesion_dynamics.copy()
    intact = block[block["lesion_kind"] == "intact"].set_index(["arm", "seed", "condition"])["state_extent"]
    keys = pd.MultiIndex.from_frame(block[["arm", "seed", "condition"]])
    block["rel_state_extent"] = block["delta_state_extent"] / keys.map(intact)
    return block[block["lesion_kind"].isin(["bidirectional", "isolation"])]


def dynamics_condition_table(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The per-condition dynamical quantities the chapter draws, as one long table (arm, seed, condition, property, value)."""
    frames = []
    props = tables["region_state_properties"]
    net = props.groupby(["arm", "seed", "condition"])[["state_extent", "state_speed", "state_pr", "cross_energy_fraction"]].mean().reset_index()
    frames.append(net.melt(id_vars=["arm", "seed", "condition"], var_name="property", value_name="value"))
    fp = tables["fixed_points_summary"]
    frames.append(fp.melt(id_vars=["arm", "seed", "condition"],
                          value_vars=["trajectory_end_speed", "nearest_distance_to_trajectory_end", "n_fixed_points"],
                          var_name="property", value_name="value"))
    jac = tables["jacobian_along_trajectory"]
    net_jac = jac[jac["scope"] == "network"].groupby(["arm", "seed", "condition"])[["n_expanding", "top_modulus"]].mean().reset_index()
    net_jac = net_jac.rename(columns={"n_expanding": "network_n_expanding", "top_modulus": "network_top_modulus"})
    frames.append(net_jac.melt(id_vars=["arm", "seed", "condition"], var_name="property", value_name="value"))
    region_jac = jac[jac["scope"] != "network"].groupby(["arm", "seed", "condition"])["top_modulus"].mean().reset_index()
    frames.append(region_jac.rename(columns={"top_modulus": "value"}).assign(property="region_top_modulus"))
    return pd.concat(frames, ignore_index=True)


def ordering_fraction(long: pd.DataFrame, *, prop: str, extreme: str = "lowest") -> pd.DataFrame:
    """Per arm: the fraction of fits in which interactive face is the lowest (or highest) on ``prop``."""
    block = long[long["property"] == prop].pivot_table(index=["arm", "seed"], columns="condition", values="value")
    pick = block.idxmin(axis=1) if extreme == "lowest" else block.idxmax(axis=1)
    out = (pick == "face_interactive").groupby(level="arm").agg(["mean", "size"]).reset_index()
    out.columns = ["arm", "fraction", "n_fits"]
    return out.assign(property=prop, extreme=extreme)


__all__ += ["dynamics_condition_table", "lesion_damage_in_r2", "lesion_extent_change", "ordering_fraction",
            "pair_lesion_share"]


# ======================================================================================
# 6. Statistics behind the chapter's figures
# ======================================================================================

#: The two contrasts every "does it cost interactive face more" panel draws.
GAP_PAIRS: tuple[tuple[str, str], ...] = (("face_interactive", "face_non_interactive"), ("face_interactive", "object"))
ALL_PAIRS: tuple[tuple[str, str], ...] = GAP_PAIRS + (("face_non_interactive", "object"),)


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm's step-down adjustment; returns adjusted p-values in the input order."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def significance_stars(p: float) -> str:
    """``***`` below 0.001, ``**`` below 0.01, ``*`` below 0.05, otherwise empty."""
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _finish_contrasts(rows: list[dict[str, object]], *, alpha: float = 0.05) -> pd.DataFrame:
    """Holm-adjust one panel's worth of tests and flag the significant ones."""
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = holm_adjust(table["p"].to_numpy())
    table["stars"] = table["p_holm"].map(significance_stars)
    table["significant"] = table["p_holm"] < alpha
    return table


def _iter_groups(frame: pd.DataFrame, by: str | Sequence[str] | None):
    if by is None:
        yield {}, frame
        return
    keys = [by] if isinstance(by, str) else list(by)
    for key, block in frame.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        yield dict(zip(keys, key)), block


def paired_contrasts(
    frame: pd.DataFrame,
    *,
    value: str,
    group: str = "condition",
    unit: str = "seed",
    by: str | Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] = ALL_PAIRS,
) -> pd.DataFrame:
    """Paired t-tests between levels of ``group``, pairing on ``unit``, separately per ``by``; Holm over the table.

    The unit is whatever is shared between the two levels -- a seed when comparing fixation
    types within one fit, a (seed, lesion) when comparing fixation types across lesions.
    """
    rows: list[dict[str, object]] = []
    for key, block in _iter_groups(frame, by):
        wide = block.pivot_table(index=unit, columns=group, values=value, aggfunc="mean")
        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns:
                continue
            both = wide[[a, b]].dropna()
            if len(both) < 3:
                continue
            t, p = _stats.ttest_rel(both[a], both[b])
            rows.append({**key, "a": a, "b": b, "n": int(len(both)), "mean_a": float(both[a].mean()), "mean_b": float(both[b].mean()),
                         "difference": float((both[a] - both[b]).mean()), "statistic": float(t), "p": float(p), "test": "paired t"})
    return _finish_contrasts(rows)


def welch_contrasts(
    frame: pd.DataFrame,
    *,
    value: str,
    group: str,
    by: str | Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Welch's t-tests between levels of ``group`` (independent samples), separately per ``by``; Holm over the table."""
    rows: list[dict[str, object]] = []
    for key, block in _iter_groups(frame, by):
        levels = list(dict.fromkeys(block[group]))
        use = pairs if pairs is not None else [(a, b) for i, a in enumerate(levels) for b in levels[i + 1:]]
        for a, b in use:
            xa = block.loc[block[group] == a, value].dropna().to_numpy(float)
            xb = block.loc[block[group] == b, value].dropna().to_numpy(float)
            if xa.size < 2 or xb.size < 2:
                continue
            t, p = _stats.ttest_ind(xa, xb, equal_var=False)
            rows.append({**key, "a": a, "b": b, "n_a": int(xa.size), "n_b": int(xb.size), "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
                         "difference": float(xa.mean() - xb.mean()), "statistic": float(t), "p": float(p), "test": "Welch t"})
    return _finish_contrasts(rows)


def gap_contrasts(
    frame: pd.DataFrame,
    *,
    value: str,
    arm: str = "arm",
    reference: str = "dense",
    group: str = "condition",
    unit: str = "seed",
    pairs: Sequence[tuple[str, str]] = GAP_PAIRS,
) -> pd.DataFrame:
    """Does a constraint widen the gap between two fixation types? Welch's t on per-seed gaps, arm against reference.

    The cost of a constraint to fixation type *a* is the dense mean minus the constrained
    value; "it costs *a* more than *b*" is the interaction, which is the per-seed difference
    ``a - b`` compared between the constrained fits and the dense fits. The seeds are
    independent between arms, so the test is Welch's. ``difference`` is the extra cost to
    *a*: the reference arm's mean gap minus the constrained arm's.
    """
    rows: list[dict[str, object]] = []
    ref = frame[frame[arm] == reference].pivot_table(index=unit, columns=group, values=value, aggfunc="mean")
    for name, block in frame[frame[arm] != reference].groupby(arm, sort=False):
        wide = block.pivot_table(index=unit, columns=group, values=value, aggfunc="mean")
        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns or a not in ref.columns or b not in ref.columns:
                continue
            gap_arm = (wide[a] - wide[b]).dropna().to_numpy(float)
            gap_ref = (ref[a] - ref[b]).dropna().to_numpy(float)
            if gap_arm.size < 2 or gap_ref.size < 2:
                continue
            t, p = _stats.ttest_ind(gap_ref, gap_arm, equal_var=False)
            rows.append({arm: name, "a": a, "b": b, "n_arm": int(gap_arm.size), "n_reference": int(gap_ref.size),
                         "gap_arm": float(gap_arm.mean()), "gap_reference": float(gap_ref.mean()),
                         "difference": float(gap_ref.mean() - gap_arm.mean()), "statistic": float(t), "p": float(p),
                         "test": "Welch t on per-seed gaps vs dense"})
    return _finish_contrasts(rows)


def one_sample_contrasts(frame: pd.DataFrame, *, value: str, by: str | Sequence[str], popmean: float) -> pd.DataFrame:
    """One-sample t-tests of ``value`` against ``popmean`` within each ``by`` group; Holm over the table."""
    rows: list[dict[str, object]] = []
    for key, block in _iter_groups(frame, by):
        x = block[value].dropna().to_numpy(float)
        if x.size < 3:
            continue
        t, p = _stats.ttest_1samp(x, popmean)
        rows.append({**key, "n": int(x.size), "mean": float(x.mean()), "popmean": float(popmean), "statistic": float(t), "p": float(p),
                     "test": f"one-sample t vs {popmean:.3g}"})
    return _finish_contrasts(rows)


def constraint_fit_long(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Every constraint's per-seed, regions-pooled fit per fixation type, with the dense fits as arm ``dense``.

    The long table the cost figures and their gap tests read: columns ``arm`` (the
    constraint name, or ``dense``), ``seed``, ``condition``, ``r2_vs_ceiling``, ``r2``.
    """
    grid = tables["grid_fit"]
    hidden = int(grid["rank_within"].max())
    arms: dict[str, pd.DataFrame] = {
        "dense": tables["dense_fit"],
        "no inter-regional connections": tables["ladder_fit"][tables["ladder_fit"]["n_partners"] == 0],
        "within-region rank 1": grid[(grid["rank_within"] == 1) & (grid["rank_cross"] == hidden)],
        "inter-regional rank 10": grid[(grid["rank_within"] == hidden) & (grid["rank_cross"] == 10)],
        "inter-regional rank 1": grid[(grid["rank_within"] == hidden) & (grid["rank_cross"] == 1)],
    }
    if "ensemble_fit" in tables:
        arms["selected: within 1, inter-regional 10"] = tables["ensemble_fit"]
    frames = []
    for name, block in arms.items():
        per = block.groupby(["seed", "condition"])[["r2_vs_ceiling", "r2"]].mean().reset_index()
        frames.append(per.assign(arm=name))
    return pd.concat(frames, ignore_index=True)


def marginal_fit_long(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-seed, regions-pooled fit per fixation type along both marginals, arm = ``<side> rank <r>``, plus ``dense``."""
    grid = tables["grid_fit"]
    hidden = int(grid["rank_within"].max())
    frames = [tables["dense_fit"].groupby(["seed", "condition"])[["r2_vs_ceiling", "r2"]].mean().reset_index()
              .assign(arm="dense", side="dense", rank=hidden)]
    for side, own, other in (("inter-regional", "rank_cross", "rank_within"), ("within-region", "rank_within", "rank_cross")):
        block = grid[grid[other] == hidden]
        for rank, part in block.groupby(own):
            if int(rank) == hidden:
                continue
            per = part.groupby(["seed", "condition"])[["r2_vs_ceiling", "r2"]].mean().reset_index()
            frames.append(per.assign(arm=f"{side} rank {int(rank)}", side=side, rank=int(rank)))
    return pd.concat(frames, ignore_index=True)


def cost_per_seed_from_long(fit_long: pd.DataFrame, *, reference: str = "dense") -> pd.DataFrame:
    """Cost of every non-reference arm per seed and fixation type: reference mean minus the arm's value."""
    ref = fit_long[fit_long["arm"] == reference].groupby("condition")["r2_vs_ceiling"].mean()
    ref_unexplained = (1.0 - fit_long[fit_long["arm"] == reference].groupby("condition")["r2"].mean())
    out = fit_long[fit_long["arm"] != reference].copy()
    out["cost"] = out["condition"].map(ref) - out["r2_vs_ceiling"]
    out["unexplained_ratio"] = (1.0 - out["r2"]) / out["condition"].map(ref_unexplained)
    return out


__all__ += ["ALL_PAIRS", "GAP_PAIRS", "constraint_fit_long", "cost_per_seed_from_long", "gap_contrasts", "holm_adjust",
            "marginal_fit_long", "one_sample_contrasts", "paired_contrasts", "significance_stars", "welch_contrasts"]


# ======================================================================================
# 7. What a lesion does to a region's local dynamics: eigenvalue spectra
# ======================================================================================


def _w2_between_spectra(a: np.ndarray, b: np.ndarray) -> float:
    """2-Wasserstein distance between two equal-size sets of complex eigenvalues (exact, by assignment)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    cost = np.abs(a[:, None] - b[None, :]) ** 2
    rows, cols = linear_sum_assignment(cost)
    return float(np.sqrt(cost[rows, cols].mean()))


def _block_spectrum(weight: np.ndarray, state_next: np.ndarray, sl: slice) -> np.ndarray:
    """Eigenvalues of a region block's local linearisation ``diag(1 - h_r(t+1)^2) W_rr``."""
    derivative = 1.0 - state_next[sl] ** 2
    return np.linalg.eigvals(derivative[:, None] * weight[sl, sl])


def _nearest_fixed_point(replay: Mapping[str, object], index: int, *, n_starts: int, iterations: int) -> tuple[np.ndarray, float, bool]:
    """The fixed point of one condition's map nearest the trajectory's end: ``(point, speed, converged)``."""
    found = find_fixed_points(replay, index, n_starts=n_starts, iterations=iterations)
    centres, _ = _cluster_points(found["points"], found["speed"], speed_tol=1e-6, merge_tol=0.1)
    end = replay["h_seq"].detach().cpu().numpy()[index, -1]
    if centres:
        distances = [float(np.linalg.norm(c - end)) for c in centres]
        k = int(np.argmin(distances))
        return np.asarray(centres[k]), 0.0, True
    slowest = int(np.argmin(found["speed"]))
    return np.asarray(found["points"][slowest]), float(found["speed"][slowest]), False


def _lesion_replays(run_dir: Path, *, kinds: Sequence[str], device: str):
    """Yield ``(kind, name, replay)`` for the intact model and every lesion of the requested kinds."""
    from dal_monte_2022_analysis.ephys.modeling.fixation_mrnn_analysis import replay_fixation_mrnn_run as _replay

    intact = _replay(run_dir, device=device)
    yield "intact", "intact", intact
    regions = list(intact["region_order"])
    checkpoint = intact["checkpoint"]
    for kind, name, blocks in _lesion_groups(regions):
        if kind not in kinds:
            continue
        replay = replay_fixation_mrnn_run_with_ablations(run_dir, ablations=list(blocks), device=device)
        for key in ("inp", "h0"):
            if key not in replay:
                replay[key] = intact[key]
        replay.setdefault("checkpoint", checkpoint)
        replay.setdefault("condition_order", intact["condition_order"])
        replay.setdefault("region_order", intact["region_order"])
        yield kind, name, replay


def lesion_region_spectra(
    run_dir: str | Path,
    *,
    kinds: Sequence[str] = ("bidirectional", "isolation"),
    time_stride: int = 5,
    network_time_stride: int = 10,
    include_network: bool = True,
    n_starts: int = 16,
    iterations: int = 1000,
    keep_spectra: bool = False,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per region block and fixation type, the eigenvalues of the lesioned network's local linearisation, against intact.

    Two linearisation points, both per fixation type: **along the trajectory** (the Jacobian
    of a region's own block at every ``time_stride``-th state the lesioned network visits,
    eigenvalues pooled over those states) and **at the fixed point** (the fixed point of
    the lesioned map nearest the end of its trajectory, found as in task 03 with fewer
    starts). For each the summary carries the largest modulus, the number of expanding
    modes and the mean modulus of the lesioned spectrum, and the 2-Wasserstein distance
    between the lesioned and the intact spectrum -- the size of the change in the region's
    local dynamics, in units of eigenvalue. With ``include_network`` the whole-network
    Jacobian is reported too, as region ``network`` (coarser ``network_time_stride`` along
    the trajectory, because the assignment behind the distance is cubic in the number of
    eigenvalues). Under a within-region rank constraint a region's own block has that many
    non-zero eigenvalues and the rest are exactly zero. Returns ``(summary, spectra)``;
    ``spectra`` holds the eigenvalues themselves only when ``keep_spectra`` is set.
    """
    run_dir = Path(run_dir)
    seed = seed_of(run_dir)
    summary_rows: list[dict[str, object]] = []
    spectra_rows: list[dict[str, object]] = []
    intact_spectra: dict[tuple[str, str, str], np.ndarray] = {}
    for kind, name, replay in _lesion_replays(run_dir, kinds=kinds, device=device):
        model = replay["model"]
        slices = model.hidden_region_slices()
        regions = list(replay["region_order"])
        conditions = [str(c) for c in replay["condition_order"]]
        weight = model.recurrent_weight_matrix().detach().cpu().numpy()
        states = replay["h_seq"].detach().cpu().numpy()
        for index, condition in enumerate(conditions):
            trajectory = states[index]
            bins = list(range(0, trajectory.shape[0] - 1, int(time_stride)))
            point, speed, converged = _nearest_fixed_point(replay, index, n_starts=n_starts, iterations=iterations)
            scopes = [(region, slices[region], bins) for region in regions]
            if include_network:
                scopes.append(("network", slice(None), list(range(0, trajectory.shape[0] - 1, int(network_time_stride)))))
            for region, sl, use_bins in scopes:
                along = np.concatenate([_block_spectrum(weight, trajectory[t + 1], sl) for t in use_bins])
                at_fp = _block_spectrum(weight, point, sl)
                for where, values in (("trajectory", along), ("fixed_point", at_fp)):
                    key = (condition, region, where)
                    if kind == "intact":
                        intact_spectra[key] = values
                    moduli = np.abs(values)
                    summary_rows.append({
                        "seed": seed, "lesion_kind": kind, "lesion": name, "condition": condition, "region": region, "where": where,
                        "n_eigenvalues": int(values.size), "top_modulus": float(moduli.max()),
                        "n_expanding": float((moduli > 1.0).sum() / (len(use_bins) if where == "trajectory" else 1)),
                        "mean_modulus": float(moduli.mean()),
                        "w2_to_intact": _w2_between_spectra(values, intact_spectra[key]) if kind != "intact" else 0.0,
                        "fixed_point_converged": bool(converged), "fixed_point_speed": float(speed),
                    })
                    if keep_spectra:
                        spectra_rows.extend({
                            "seed": seed, "lesion_kind": kind, "lesion": name, "condition": condition, "region": region, "where": where,
                            "re": float(v.real), "im": float(v.imag),
                        } for v in values)
    return pd.DataFrame(summary_rows), pd.DataFrame(spectra_rows)


def collect_lesion_spectra(arm_dirs: Mapping[str, str | Path], *, device: str = "cpu", **kwargs) -> pd.DataFrame:
    """:func:`lesion_region_spectra` summaries over every seed of every arm."""
    frames = []
    for arm, path in arm_dirs.items():
        for run_dir in seed_run_dirs(path):
            summary, _ = lesion_region_spectra(run_dir, device=device, **kwargs)
            frames.append(summary.assign(arm=arm))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


__all__ += ["collect_lesion_spectra", "lesion_region_spectra"]


def lesion_damage_per_fit(lesion_battery: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, fit, lesion family): mean damage over that family's lesions, beside the size-matched random control.

    Damage is ``damage_shared_r2`` summed over the twelve region × fixation-type
    combinations -- the drop in pooled R^2 -- averaged over the lesions of a family. The
    control is the mean over the random draws that silenced the same number of weights as
    that family's lesions, in the same fit, so the two are paired by fit.
    """
    per_lesion = (lesion_battery.groupby(["arm", "seed", "lesion_kind", "lesion", "live_weights_removed"])["damage_shared_r2"]
                  .sum().reset_index())
    controls = per_lesion[per_lesion["lesion_kind"] == "random control"]
    control_by_size = controls.groupby(["arm", "seed", "live_weights_removed"])["damage_shared_r2"].mean()
    families = per_lesion[per_lesion["lesion_kind"] != "random control"]
    out = families.groupby(["arm", "seed", "lesion_kind"]).agg(damage=("damage_shared_r2", "mean"),
                                                               live_weights_removed=("live_weights_removed", "first")).reset_index()
    out["control"] = [float(control_by_size.get((a, s, int(n)), np.nan)) for a, s, n in
                      zip(out["arm"], out["seed"], out["live_weights_removed"])]
    return out.rename(columns={"lesion_kind": "family"})


__all__ += ["lesion_damage_per_fit"]
