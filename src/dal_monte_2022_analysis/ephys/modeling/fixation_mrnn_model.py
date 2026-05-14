"""Installed-mrnntorch fixation mRNN model wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FixationMRNNModelSpec:
    """Serializable model construction settings."""

    canonical_region_order: tuple[str, ...]
    internal_region_order: tuple[str, ...]
    hidden_units_by_region: dict[str, int]
    output_dims_by_region: dict[str, int]
    input_dim: int = 3
    input_region_name: str = "input"
    activation: str = "softplus"
    dt: float = 10.0
    tau: float = 100.0
    inp_noise: float = 0.0
    act_noise: float = 0.0
    rec_constrained: bool = False
    inp_constrained: bool = False
    batch_first: bool = True
    spectral_radius: float | None = 1.3
    device: str = "cpu"


class FixationMRNNModel(nn.Module):
    """Multi-region mRNN with per-region readouts for fixation PSTH targets."""

    def __init__(self, spec: FixationMRNNModelSpec):
        super().__init__()
        self.spec = spec
        self.canonical_region_order = tuple(spec.canonical_region_order)
        self.internal_region_order = tuple(spec.internal_region_order)

        if set(self.canonical_region_order) != set(self.internal_region_order):
            raise ValueError(
                "internal_region_order must contain the same regions as "
                "canonical_region_order."
            )

        try:
            from mrnntorch import mRNN
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The installed lowercase 'mrnntorch' package is required for "
                "fixation mRNN modeling."
            ) from exc

        self.mrnn = mRNN(
            activation=spec.activation,
            noise_level_act=float(spec.act_noise),
            noise_level_inp=float(spec.inp_noise),
            rec_constrained=bool(spec.rec_constrained),
            inp_constrained=bool(spec.inp_constrained),
            batch_first=bool(spec.batch_first),
            spectral_radius=spec.spectral_radius,
            device=str(spec.device),
            dt=float(spec.dt),
            tau=float(spec.tau),
        )
        for region in self.internal_region_order:
            self.mrnn.add_recurrent_region(
                region,
                int(spec.hidden_units_by_region[region]),
                sign="pos",
                device=str(spec.device),
                learnable_bias=True,
            )
        self.mrnn.add_input_region(
            spec.input_region_name,
            int(spec.input_dim),
            sign="pos",
            device=str(spec.device),
        )
        for src_region in self.internal_region_order:
            for dst_region in self.internal_region_order:
                self.mrnn.add_recurrent_connection(src_region, dst_region, sparsity=1.0)
        for dst_region in self.internal_region_order:
            self.mrnn.add_input_connection(
                spec.input_region_name,
                dst_region,
                sparsity=1.0,
            )
        self.mrnn.finalize_connectivity()

        self.output_heads = nn.ModuleDict(
            {
                region: nn.Linear(
                    int(spec.hidden_units_by_region[region]),
                    int(spec.output_dims_by_region[region]),
                )
                for region in self.canonical_region_order
            }
        )

    @property
    def total_num_units(self) -> int:
        """Return the total recurrent state size."""
        return int(self.mrnn.total_num_units)

    def forward(
        self,
        inp: torch.Tensor,
        x0: torch.Tensor,
        h0: torch.Tensor | None = None,
        *,
        stim_input: torch.Tensor | None = None,
        noise: bool = True,
    ) -> dict[str, object]:
        """Run the mRNN and return per-region and concatenated outputs."""
        x_seq, h_seq = self.mrnn(
            inp,
            x0,
            h0,
            stim_input=stim_input,
            noise=bool(noise),
        )
        output_by_region = {}
        for region in self.canonical_region_order:
            region_h = self.mrnn.get_region_activity(h_seq, region)
            output_by_region[region] = self.output_heads[region](region_h)
        output = torch.cat(
            [output_by_region[region] for region in self.canonical_region_order],
            dim=-1,
        )
        return {
            "output": output,
            "output_by_region": output_by_region,
            "x_seq": x_seq,
            "h_seq": h_seq,
        }


def build_model_spec(
    *,
    canonical_region_order: tuple[str, ...],
    internal_region_order: tuple[str, ...],
    output_dims_by_region: dict[str, int],
    hidden_units: int | dict[str, int] = 100,
    device: str = "cpu",
    **kwargs,
) -> FixationMRNNModelSpec:
    """Build a model spec from scalar or per-region hidden-unit settings."""
    if isinstance(hidden_units, dict):
        hidden_units_by_region = {
            region: int(hidden_units[region]) for region in canonical_region_order
        }
    else:
        hidden_units_by_region = {
            region: int(hidden_units) for region in canonical_region_order
        }
    return FixationMRNNModelSpec(
        canonical_region_order=tuple(canonical_region_order),
        internal_region_order=tuple(internal_region_order),
        hidden_units_by_region=hidden_units_by_region,
        output_dims_by_region={
            region: int(output_dims_by_region[region])
            for region in canonical_region_order
        },
        device=str(device),
        **kwargs,
    )


__all__ = [
    "FixationMRNNModel",
    "FixationMRNNModelSpec",
    "build_model_spec",
]
