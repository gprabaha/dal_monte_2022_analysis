import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mrnntorch.mRNN import mRNN


class Model(nn.Module):
    def __init__(
        self,
        config,
        hid_dim,
        pfc_units,
        acc_units,
        ofc_units,
        bla_units,
        dt,
        tau,
        inp_noise,
        act_noise,
        rec_constrained,
        inp_constrained,
        batch_first,
        spectral_radius,
        activation="softplus",
        output_layer=True,
        latent_training=False,
        n_components=10,
        device="cuda",
    ):
        super(Model, self).__init__()

        self.hid_dim = hid_dim
        self.dt = dt
        self.tau = tau
        self.inp_noise = inp_noise
        self.act_noise = act_noise
        self.output_layer = output_layer
        self.latent_training = latent_training
        self.n_components = n_components

        if self.output_layer:
            self.mrnn = mRNN(
                config,
                activation=activation,
                rec_constrained=rec_constrained,
                inp_constrained=inp_constrained,
                batch_first=batch_first,
                dt=dt,
                tau=tau,
                noise_level_act=act_noise,
                noise_level_inp=inp_noise,
                spectral_radius=spectral_radius,
                device=device,
            )
        else:
            self.mrnn = mRNN(
                activation=activation,
                rec_constrained=rec_constrained,
                inp_constrained=inp_constrained,
                batch_first=batch_first,
                dt=dt,
                tau=tau,
                noise_level_act=act_noise,
                noise_level_inp=inp_noise,
                spectral_radius=spectral_radius,
                device=device,
            )

        self.connection_props = ["pfc", "acc", "ofc", "bla"]

        self.region_units = {
            "pfc": pfc_units,
            "acc": acc_units,
            "ofc": ofc_units,
            "bla": bla_units,
        }

        # If using output layer, define regions in config, otherwise here
        if not self.output_layer:
            # Define all recurrent regions
            for region in self.connection_props:
                self.mrnn.add_recurrent_region(region, self.region_units[region])

            # Define input region
            self.mrnn.add_input_region("input", 3)

            # Add input connections
            for region in self.connection_props:
                self.mrnn.add_input_connection("input", region)
        else:
            if latent_training:
                self.pfc_out = nn.Linear(hid_dim, n_components)
                self.acc_out = nn.Linear(hid_dim, n_components)
                self.ofc_out = nn.Linear(hid_dim, n_components)
                self.bla_out = nn.Linear(hid_dim, n_components)
            else:
                self.pfc_out = nn.Linear(hid_dim, pfc_units)
                self.acc_out = nn.Linear(hid_dim, acc_units)
                self.ofc_out = nn.Linear(hid_dim, ofc_units)
                self.bla_out = nn.Linear(hid_dim, bla_units)

        # Build fully connected network with proper cell types
        for src_region in self.connection_props:
            for dst_region in self.connection_props:
                self.mrnn.add_recurrent_connection(src_region, dst_region)
        self.mrnn.finalize_connectivity()

        self.out_order = ["ofc", "bla", "pfc", "acc"]

    def forward(self, xn, inp, *args, noise=True):
        xn, hn = self.mrnn(xn, inp, *args, noise=noise)

        if self.output_layer:
            pfc_act = self.mrnn.get_region_activity(hn, "pfc")
            acc_act = self.mrnn.get_region_activity(hn, "acc")
            ofc_act = self.mrnn.get_region_activity(hn, "ofc")
            bla_act = self.mrnn.get_region_activity(hn, "bla")

            pfc_out = self.pfc_out(pfc_act)
            acc_out = self.acc_out(acc_act)
            ofc_out = self.ofc_out(ofc_act)
            bla_out = self.bla_out(bla_act)

            out = torch.cat([ofc_out, bla_out, pfc_out, acc_out], dim=-1)

        else:
            out = hn

        return out, hn
