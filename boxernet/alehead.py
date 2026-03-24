import numpy as np
import torch
import torch.nn as nn
from utils.obb import ObbTW
from utils.pose import PoseTW, rotation_from_euler


class AleHead(torch.nn.Module):
    """Aleatoric uncertainty head. Predicts 3D bounding boxes."""

    def __init__(self, in_dim, out_dim=7, hidden_dim=128, norm_chamfer=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bbox_min = 0.02
        self.bbox_max = 4.0
        self.norm_chamfer = norm_chamfer
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, out_dim)
        self.logvar_head = nn.Linear(hidden_dim, 1)  # predicts log σ²

    def forward(self, batch, output):
        query = output["query"]

        h = self.net(query)
        mu = self.mean_head(h)
        logvar = self.logvar_head(h)
        logvar = torch.clamp(logvar, -10, +3)

        params_v = mu

        B, M = params_v.shape[0], params_v.shape[1]
        device = params_v.device
        pr = params_v
        yaw_max = np.pi / 2

        # Directly predict X,Y,Z in voxel coords.
        center_v = pr[..., :3]

        # Build size
        pr[..., 3:6] = (
            torch.sigmoid(pr[..., 3:6]) * (self.bbox_max - self.bbox_min)
            + self.bbox_min
        )
        bb3 = torch.zeros((B, M, 6)).to(device)
        hh, ww, dd = pr[:, :, 3], pr[:, :, 4], pr[:, :, 5]
        bb3[:, :, 0] = -(hh / 2)
        bb3[:, :, 1] = hh / 2
        bb3[:, :, 2] = -(ww / 2)
        bb3[:, :, 3] = ww / 2
        bb3[:, :, 4] = -(dd / 2)
        bb3[:, :, 5] = dd / 2

        # Build T_world_object
        pr[..., 6] = yaw_max * torch.tanh(pr[..., 6])
        yaw = pr[:, :, 6]
        zeros = torch.zeros_like(yaw).to(device)
        e_angles = torch.stack([zeros, zeros, yaw], dim=-1)
        R = rotation_from_euler(e_angles.reshape(-1, 3))
        R = R.reshape(B, M, 3, 3)
        T_vo = PoseTW.from_Rt(R, center_v)

        # Build final ObbTW.
        # prob = torch.ones((B, M, 1)).to(device)
        sigma2 = torch.exp(logvar)
        prob = 1.0 / (1.0 + sigma2)

        inst_id = torch.arange(M).reshape(1, M, -1).repeat(B, 1, 1).to(device)
        sem_id = 32 + torch.zeros((B, M, 1)).to(device)
        obb_pr_v = ObbTW.from_lmc(
            bb3_object=bb3,
            T_world_object=T_vo,
            prob=prob,
            inst_id=inst_id,
            sem_id=sem_id,
        )
        # obb_pr_v._data[~valid] = -1  # TODO(dd): adding this removes too many boxes?
        T_wv = batch["T_world_voxel0"].unsqueeze(1)
        obb_pr_w = obb_pr_v.transform(T_wv)
        output["obbs_pr_w"] = obb_pr_w
        output["obbs_pr_params"] = params_v
        output["obbs_pr_logvar"] = logvar
        return batch, output
