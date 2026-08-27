"""Minimal Elman mRNN model for fixation PSTH targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FixationMRNNModelSpec:
    """Everything needed to construct the model."""

    region_order: tuple[str, ...]
    hidden_units_by_region: dict[str, int]
    output_dims_by_region: dict[str, int]
    input_dim: int = 3
    input_region_name: str = "input"
    activation: str = "softplus"
    spectral_radius: float | None = 1.3
    rec_constrained: bool = False
    inp_constrained: bool = False
    recurrent_connectivity: str = "full"
    recurrent_bottleneck_dim: int = 10
    batch_first: bool = True
    inp_noise: float = 0.0
    act_noise: float = 0.0
    device: str = "cpu"


class FixationMRNNModel(nn.Module):
    """Elman mRNN wrapper with region-based readouts and custom recurrent parameterization.

    This class wraps an underlying `mrnntorch.ElmanmRNN` instance in `self.mrnn`.
    The wrapper uses `mrnntorch` for region registration, connectivity topology,
    activation functions, input connectivity, and device placement, but it does not
    rely on `self.mrnn.forward(...)` for the recurrence computation.

    Instead, `FixationMRNNModel` constructs its own recurrent weight matrix from:
    - within-region parameter blocks stored in `self._within_region_params`
    - low-rank inter-region factors stored in `self._inter_region_left_params`
      and `self._inter_region_right_params`

    The full dense recurrent matrix is assembled in `_build_recurrent_weight_and_mask()`
    and synchronized into the `mrnntorch` submodule via `_sync_recurrent_state()`
    before the wrapper's own `forward()` loop executes.
    """

    def __init__(self, spec: FixationMRNNModelSpec):
        super().__init__()
        self.spec = spec
        self.region_order = tuple(spec.region_order)
        self._connectivity_mode = normalize_recurrent_connectivity(spec.recurrent_connectivity)
        self._bottleneck_dim = int(spec.recurrent_bottleneck_dim)
        self._within_region_params: dict[str, nn.Parameter] = {}
        self._inter_region_left_params: dict[tuple[str, str], nn.Parameter] = {}
        self._inter_region_right_params: dict[tuple[str, str], nn.Parameter] = {}
        try:
            from mrnntorch import ElmanmRNN
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("The lowercase 'mrnntorch' package is required.") from exc

        self.mrnn = ElmanmRNN(
            activation=spec.activation,
            noise_level_act=float(spec.act_noise),
            noise_level_inp=float(spec.inp_noise),
            rec_constrained=bool(spec.rec_constrained),
            inp_constrained=bool(spec.inp_constrained),
            batch_first=bool(spec.batch_first),
            spectral_radius=spec.spectral_radius,
            config_finalize=False,
            device=str(spec.device),
        )
        # `mrnntorch.ElmanmRNN` handles the region definitions, input/output
        # connectivity metadata, and device-aware tensor allocation.
        # Here we register regions and connectivity, then finalize the internal
        # connectivity graph before creating our own recurrent parameters.
        for region in self.region_order:
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
        for source in self.region_order:
            for target in self.region_order:
                if _region_pair_connected(source, target, spec.recurrent_connectivity):
                    self.mrnn.add_recurrent_connection(source, target)
        for target in self.region_order:
            self.mrnn.add_input_connection(spec.input_region_name, target)
        self.mrnn.finalize_connectivity()
        self._initialize_recurrent_parameters(spec)
        self._sync_recurrent_state()

        self.output_heads = nn.ModuleDict(
            {
                region: nn.Linear(
                    int(spec.hidden_units_by_region[region]),
                    int(spec.output_dims_by_region[region]),
                )
                for region in self.region_order
            }
        )

    @property
    def total_num_units(self) -> int:
        return int(self.mrnn.total_num_units)

    def hidden_region_slices(self) -> dict[str, slice]:
        """Return hidden-state slices for each recurrent region."""
        slices = {}
        for region in self.region_order:
            start, stop = self.mrnn.get_region_indices(region)
            slices[region] = slice(int(start), int(stop))
        return slices

    def _initialize_recurrent_parameters(self, spec: FixationMRNNModelSpec) -> None:
        """Create trainable parameters for each allowed recurrent block.

        This wrapper uses two distinct parameter types:
        - within-region parameters stored in `self._within_region_params`
          for local recurrent coupling inside each region.
        - low-rank inter-region factors stored in
          `self._inter_region_left_params` and `self._inter_region_right_params`
          for cross-region recurrence.

        The wrapper then registers these tensors as parameters so they appear
        in `model.parameters()` and can be optimized by PyTorch.
        """
        for region in self.region_order:
            hidden_units = int(spec.hidden_units_by_region[region])
            if self._connectivity_mode == "cross_region_with_self_diagonal":
                # In the self-diagonal connectivity mode, only the diagonal entries
                # of each within-region block are used.
                parameter = nn.Parameter(torch.zeros(hidden_units, dtype=torch.float32))
                nn.init.uniform_(parameter, -0.1, 0.1)
            else:
                # In the full connectivity mode, each within-region block is a
                # full dense matrix of shape (hidden_units, hidden_units).
                parameter = nn.Parameter(torch.zeros(hidden_units, hidden_units, dtype=torch.float32))
                nn.init.xavier_uniform_(parameter)
            self._within_region_params[region] = parameter
            self.register_parameter(f"_within_region_param_{region}", parameter)

        for source in self.region_order:
            for target in self.region_order:
                if source == target or not _region_pair_connected(source, target, spec.recurrent_connectivity):
                    continue
                source_units = int(spec.hidden_units_by_region[source])
                target_units = int(spec.hidden_units_by_region[target])
                left = nn.Parameter(torch.empty(source_units, self._bottleneck_dim, dtype=torch.float32))
                right = nn.Parameter(torch.empty(self._bottleneck_dim, target_units, dtype=torch.float32))
                nn.init.xavier_uniform_(left)
                nn.init.xavier_uniform_(right)
                # Inter-region recurrence is factorized as left @ right,
                # which enforces a rank constraint on cross-region connections.
                self._inter_region_left_params[(source, target)] = left
                self._inter_region_right_params[(source, target)] = right
                self.register_parameter(f"_inter_left_{source}_{target}", left)
                self.register_parameter(f"_inter_right_{source}_{target}", right)

    def _within_region_block(self, region: str) -> torch.Tensor:
        parameter = self._within_region_params[region]
        if self._connectivity_mode == "cross_region_with_self_diagonal":
            return torch.diag(parameter)
        return parameter

    def _inter_region_block(self, source: str, target: str) -> torch.Tensor:
        return self._inter_region_left_params[(source, target)] @ self._inter_region_right_params[(source, target)]

    def _build_recurrent_weight_and_mask(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble the full dense recurrent weight matrix and its mask/sign tensors.

        The weight matrix is arranged in region-ordered blocks. For each source/target
        region pair we select one of:
        - a within-region block from `self._within_region_params`
        - a low-rank inter-region block from `self._inter_region_left_params` and
          `self._inter_region_right_params`
        - a zero block when no connectivity exists

        The mask tensor indicates which entries are structurally present, and the
        sign tensor is currently a placeholder of all ones for compatibility with
        `mrnntorch`'s recurrent weight conventions.
        """
        hidden_slices = self.hidden_region_slices()
        total_num_units = self.total_num_units
        weight = torch.zeros(total_num_units, total_num_units, dtype=torch.float32, device=self.mrnn.device)
        mask = torch.zeros(total_num_units, total_num_units, dtype=torch.float32, device=self.mrnn.device)
        sign = torch.ones(total_num_units, total_num_units, dtype=torch.float32, device=self.mrnn.device)

        for target in self.region_order:
            target_slice = hidden_slices[target]
            for source in self.region_order:
                source_slice = hidden_slices[source]
                if source == target:
                    block = self._within_region_block(target)
                    block_mask = torch.ones_like(block)
                    if self._connectivity_mode == "cross_region_with_self_diagonal":
                        # In cross-region-only mode, the within-region block is
                        # restricted to a diagonal matrix.
                        block_mask = torch.eye(block.shape[0], device=block.device, dtype=block_mask.dtype)
                elif _region_pair_connected(source, target, self.spec.recurrent_connectivity):
                    block = self._inter_region_block(source, target)
                    block_mask = torch.ones_like(block)
                else:
                    block = torch.zeros(
                        int(self.spec.hidden_units_by_region[target]),
                        int(self.spec.hidden_units_by_region[source]),
                        dtype=torch.float32,
                        device=self.mrnn.device,
                    )
                    block_mask = torch.zeros_like(block)

                # Write the selected block into the full dense matrix.
                weight[target_slice, source_slice] = block
                mask[target_slice, source_slice] = block_mask

        return weight, mask, sign

    def _sync_recurrent_state(self) -> None:
        """Synchronize the wrapper's assembled weight matrix with the mrnntorch submodule.

        This method rebuilds the dense recurrent weight matrix from the current
        parameter blocks, then assigns it into `self.mrnn.W_rec` so that the
        underlying `mrnntorch` module sees the same recurrent weights.
        The mask and sign tensors are also updated for compatibility.
        """
        weight, mask, sign = self._build_recurrent_weight_and_mask()
        self.mrnn.W_rec = nn.Parameter(weight.detach(), requires_grad=True)
        self.mrnn.W_rec_mask = nn.Parameter(mask.detach(), requires_grad=False)
        self.mrnn.W_rec_sign_matrix = nn.Parameter(sign.detach(), requires_grad=False)

    def _enable_self_diagonal_recurrent_mask(self) -> None:
        """Keep the compatibility hook for self-diagonal recurrent masking."""
        self._sync_recurrent_state()

    def output_region_slices(self) -> dict[str, slice]:
        """Return concatenated-output slices for each readout region."""
        slices = {}
        start = 0
        for region in self.region_order:
            stop = start + int(self.spec.output_dims_by_region[region])
            slices[region] = slice(start, stop)
            start = stop
        return slices

    def readout_weight_matrix(self) -> torch.Tensor:
        """Return the explicit block-local hidden-to-output readout matrix.

        The forward path uses one linear head per region. This helper exposes
        the equivalent concatenated matrix, with zeros outside each
        region's own hidden-to-output block.
        """
        first_weight = next(iter(self.output_heads.values())).weight
        weight = torch.zeros(
            sum(int(self.spec.output_dims_by_region[region]) for region in self.region_order),
            self.total_num_units,
            dtype=first_weight.dtype,
            device=first_weight.device,
        )
        hidden_slices = self.hidden_region_slices()
        output_slices = self.output_region_slices()
        for region in self.region_order:
            weight[output_slices[region], hidden_slices[region]] = self.output_heads[region].weight
        return weight

    def readout_bias_vector(self) -> torch.Tensor:
        """Return the concatenated output bias ordered by region."""
        return torch.cat([self.output_heads[region].bias for region in self.region_order], dim=0)

    def recurrent_weight_matrix(self) -> torch.Tensor:
        """Return the explicit dense recurrent weight matrix used by the forward pass.

        This helper rebuilds the current matrix from the wrapper-managed parameter
        blocks and re-syncs the `mrnntorch` submodule state. The returned weight is
        the same dense recurrent matrix that will be used in `forward()`.
        """
        weight, _, _ = self._build_recurrent_weight_and_mask()
        self._sync_recurrent_state()
        return weight

    def within_region_recurrent_parameters(self) -> list[nn.Parameter]:
        """Return the trainable within-region recurrent parameters."""
        return [self._within_region_params[region] for region in self.region_order]

    def inter_region_recurrent_parameters(self) -> list[tuple[nn.Parameter, nn.Parameter]]:
        """Return the trainable low-rank factors for inter-region recurrent connections."""
        return [
            (self._inter_region_left_params[(source, target)], self._inter_region_right_params[(source, target)])
            for source in self.region_order
            for target in self.region_order
            if source != target and (source, target) in self._inter_region_left_params
        ]

    def within_region_recurrent_l1_penalty(self, *, scale: float) -> torch.Tensor:
        """Apply L1 regularization only to within-region recurrent weights."""
        if float(scale) <= 0.0:
            return torch.zeros((), device=next(iter(self._within_region_params.values())).device)
        penalty = torch.zeros((), device=next(iter(self._within_region_params.values())).device)
        for parameter in self.within_region_recurrent_parameters():
            penalty = penalty + torch.mean(torch.abs(parameter))
        return penalty * float(scale)

    def forward(
        self,
        inp: torch.Tensor,
        h0: torch.Tensor,
        *,
        stim_input: torch.Tensor | None = None,
        noise: bool = False,
    ) -> dict[str, object]:
        """Execute the wrapper's custom recurrent forward pass.

        This method is the active forward pass for `FixationMRNNModel`.
        It does not call `self.mrnn.forward(...)`; instead it manually computes the
        recurrent updates using the dense matrix assembled from our custom blocks.

        Execution order:
        1. Sync the current block-parameter state into `self.mrnn`.
        2. Build or fetch the dense recurrent weight matrix.
        3. Compute input weights via `self.mrnn.W_inp` and optional Dales law.
        4. Iterate through time, applying recurrent and input contributions.
        5. Apply `self.mrnn.activation(...)` to produce the next hidden state.
        6. Assemble outputs region-by-region using the linear readout heads.
        """
        self._sync_recurrent_state()
        recurrent_weight = self.recurrent_weight_matrix()
        if self.mrnn.inp_constrained:
            w_inp = self.mrnn.apply_dales_law(self.mrnn.W_inp, self.mrnn.W_inp_mask, self.mrnn.W_inp_sign_matrix)
        else:
            w_inp = self.mrnn.W_inp * self.mrnn.W_inp_mask
        baseline_inp = self.mrnn.tonic_inp

        if self.spec.batch_first:
            batch_shape = inp.shape[0]
            seq_len = inp.shape[1]
            shape = (batch_shape, seq_len, self.total_num_units)
        else:
            seq_len = inp.shape[0]
            batch_shape = inp.shape[1]
            shape = (seq_len, batch_shape, self.total_num_units)

        new_hs = torch.empty(size=shape, device=inp.device, dtype=inp.dtype)
        hn_next = h0
        for t in range(seq_len):
            if self.spec.batch_first:
                inp_t = inp[:, t, :]
            else:
                inp_t = inp[t, :, :]
            if noise:
                hid_noise = self.mrnn._hid_noise(batch_shape)
                inp_noise = self.mrnn._inp_noise(batch_shape)
            else:
                hid_noise = inp_noise = 0
            xn_next = (
                +(recurrent_weight @ hn_next.T).T
                + (w_inp @ (inp_t + inp_noise).T).T
                + baseline_inp
                + hid_noise
            )
            if stim_input is not None:
                if self.spec.batch_first:
                    xn_next = xn_next + stim_input[:, t, :]
                else:
                    xn_next = xn_next + stim_input[t, :, :]
            # Activation is delegated to the underlying mrnntorch module.
            hn_next = self.mrnn.activation(xn_next)
            if self.spec.batch_first:
                new_hs[:, t, :] = hn_next
            else:
                new_hs[t, :, :] = hn_next

        h_seq = new_hs
        output_by_region = {}
        for region in self.region_order:
            region_h = self.mrnn.get_region_activity(h_seq, region)
            output_by_region[region] = self.output_heads[region](region_h)
        output = torch.cat([output_by_region[region] for region in self.region_order], dim=-1)
        return {
            "output": output,
            "output_by_region": output_by_region,
            "h_seq": h_seq,
        }


def build_model_spec(
    *,
    region_order: tuple[str, ...],
    output_dims_by_region: dict[str, int],
    hidden_units: int | dict[str, int],
    device: str,
    **kwargs,
) -> FixationMRNNModelSpec:
    """Build a spec, expanding scalar hidden units across regions."""
    if isinstance(hidden_units, dict):
        hidden_by_region = {region: int(hidden_units[region]) for region in region_order}
    else:
        hidden_by_region = {region: int(hidden_units) for region in region_order}
    return FixationMRNNModelSpec(
        region_order=tuple(region_order),
        hidden_units_by_region=hidden_by_region,
        output_dims_by_region={region: int(output_dims_by_region[region]) for region in region_order},
        device=str(device),
        **kwargs,
    )


RECURRENT_CONNECTIVITY_ALIASES = {
    "full": "full",
    "all": "full",
    "all_region": "full",
    "all_region_to_region": "full",
    "all_regions": "full",
    "within": "within_region",
    "within_region": "within_region",
    "internal": "within_region",
    "internal_only": "within_region",
    "cross": "cross_region_with_self_diagonal",
    "cross_region": "cross_region_with_self_diagonal",
    "cross_region_only": "cross_region_with_self_diagonal",
    "cross_region_with_self_diagonal": "cross_region_with_self_diagonal",
    "cross_plus_diagonal": "cross_region_with_self_diagonal",
}


def normalize_recurrent_connectivity(connectivity: str) -> str:
    """Normalize recurrent connectivity aliases."""
    token = str(connectivity).strip().lower().replace("-", "_")
    try:
        return RECURRENT_CONNECTIVITY_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "recurrent_connectivity must be one of: "
            "'full', 'within_region', or 'cross_region_with_self_diagonal'."
        ) from exc


def _region_pair_connected(source: str, target: str, connectivity: str) -> bool:
    mode = normalize_recurrent_connectivity(connectivity)
    if mode == "full":
        return True
    if mode == "within_region":
        return source == target
    if mode == "cross_region_with_self_diagonal":
        return source != target
    raise ValueError(f"Unsupported recurrent connectivity: {connectivity!r}")


__all__ = [
    "FixationMRNNModel",
    "FixationMRNNModelSpec",
    "build_model_spec",
    "normalize_recurrent_connectivity",
]
