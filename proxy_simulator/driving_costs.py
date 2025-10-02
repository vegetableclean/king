import torch

import numpy as np


class RouteDeviationCostRasterized:
    """
    Measures overlap between an agent and the non-drivable area
    using Gaussian kernels around the vehicle corners.
    """
    def __init__(self, sim_args):
        self.batch_size = sim_args.batch_size
        self.num_agents = sim_args.num_agents

        # Resolve a safe device from sim_args
        self.device = torch.device(sim_args.device) if isinstance(sim_args.device, str) else sim_args.device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        # constants
        self.sigma_x = 5 * 128
        self.sigma_y = 2 * 128
        self.variance_x = self.sigma_x ** 2.0
        self.variance_y = self.sigma_y ** 2.0

        # store constants on self.device; will be moved to input device at use
        self.PI = torch.tensor(np.pi, device=self.device, dtype=torch.float32)
        self.original_corners = torch.tensor(
            [[ 1.0,  2.5],
             [ 1.0, -2.5],
             [-1.0,  2.5],
             [-1.0, -2.5]],
            device=self.device, dtype=torch.float32
        )

    def get_corners(self, pos, yaw):
        """
        Given ego/adv positions and yaw, return corners as shape [1, NumCorners, 2]
        pos: [B, N, 2], yaw: [B, N, 1] (B should be 1 here)
        """
        device = yaw.device
        PI = self.PI.to(device)
        corners0 = self.original_corners.to(device)

        # rotate so that heading 0 points "up"
        yaw = PI / 2.0 - yaw  # [B, N, 1]

        rot_mat = torch.cat(
            [torch.cos(yaw), -torch.sin(yaw),
             torch.sin(yaw),  torch.cos(yaw)],
            dim=-1,
        ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2)  # [N, 4, 2, 2]

        # rotate canonical corners and translate to agent pos
        rotated = rot_mat @ corners0.unsqueeze(-1)             # [N, 4, 2, 1]
        rotated = rotated.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)  # [N, 4, 2]

        return rotated.view(1, -1, 2)  # [1, N*4, 2]

    def crop_map(self, j, i, y_extent, x_extent, road_rasterized):
        """
        Crop a 64x64 window (with 2px stride → 32x32) centered at (i,j).
        """
        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))
        return road_rasterized[i_min:i_max:2, j_min:j_max:2]

    def get_pixel_grid(self, i, j, x_extent, y_extent, *, device):
        """
        Return coords grid with shape [H/2, W/2, 2] on `device`
        """
        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))

        yy = torch.linspace(i_min, i_max - 1, (i_max - i_min), device=device)
        xx = torch.linspace(j_min, j_max - 1, (j_max - j_min), device=device)
        yy, xx = torch.meshgrid(yy, xx)  # PyTorch 1.12 default indexing
        coords = torch.stack([yy, xx], dim=-1)  # [H, W, 2]
        coords = coords[::2, ::2].float()      # stride 2
        return coords

    def apply_gauss_kernels(self, coords_list, pos):
        """
        coords_list: list of [1, H, W, 2] (each on device)
        pos: [B, NumCorners, 2]
        returns: [NumCorners, H, W] on the same device
        """
        sigma = 5.0
        device = pos.device
        pos2d = pos[0, :, :]  # [NumCorners, 2]
        coords = torch.cat(coords_list, dim=0).to(device)  # [NumCorners, H, W, 2]

        gk = torch.mean(((coords - pos2d[:, None, None, :]) / sigma) ** 2, dim=-1)  # [NumCorners, H, W]
        gk = (1.0 / (2.0 * self.PI.to(device) * sigma * sigma)) * torch.exp(-gk)
        return gk

    def __call__(self, road_rasterized, pos, yaw, crop_center, pos_w2m):
        """
        road_rasterized: [H, W] binary road mask (1 = road)
        pos, yaw: ego/adv states
        crop_center: [2]
        pos_w2m: world→map transform callable
        """
        device = road_rasterized.device
        pos = pos.to(device)
        yaw = yaw.to(device)

        corners_pos = self.get_corners(pos, yaw)                    # [1, NumCorners, 2] on device
        crop_center = pos_w2m(crop_center[None].to(device))[0]      # [2] on device
        corners_pos = pos_w2m(corners_pos.view(-1, 2)).view(self.batch_size, self.num_agents * 4, 2)

        x_extent, y_extent = road_rasterized.size(0), road_rasterized.size(1)

        # crop road around each corner (in map pixels)
        crops = []
        for k in range(corners_pos.size(1)):
            c = corners_pos[0, k, :]  # [2]
            crop = self.crop_map(
                c[0].item(), c[1].item(),
                x_extent, y_extent,
                road_rasterized
            )
            crop = (1.0 - crop).to(device)  # want non-drivable
            crops.append(crop)

        # coords for each crop
        coords_list = []
        for k in range(corners_pos.size(1)):
            c = corners_pos[0, k, :]
            coords_list.append(
                self.get_pixel_grid(
                    c[0].item(), c[1].item(),
                    x_extent, y_extent,
                    device=device
                ).unsqueeze(0)  # [1, H, W, 2]
            )

        gks = self.apply_gauss_kernels(coords_list, corners_pos)    # [NumCorners, H, W]
        roads_rasterized = torch.stack(crops, dim=0)                # [NumCorners, H, W]

        # sum over corners and spatial dims → scalar cost (keep [1,1] shape for callers)
        cost_scalar = (roads_rasterized * gks).sum()
        return cost_scalar.view(1, 1)  # [1,1]


class BatchedPolygonCollisionCost():
    """
    """
    def __init__(self, sim_args):
        """
        """
        self.sim_args = sim_args
        self.batch_size = sim_args.batch_size
        self.num_agents = sim_args.num_agents

        self.unit_square = torch.tensor(
            [[ 1.,  1.],
             [-1.,  1.],
             [-1., -1.],
             [ 1., -1.]], dtype=torch.float32
        ).view(1, 1, 4, 2)  # keep on CPU; move to device at use time
        # moved at call time: .to(device).expand(batch, num_agents+1, 4, 2)

        self.segment_start_transform = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
            
        ).reshape(1, 4, 4)

        self.segment_end_transform = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=torch.float32,
            
        ).reshape(1, 4, 4)

    def vertices_to_edges_vectorized(self, vertices):
        device = vertices.device
        seg_start = self.segment_start_transform.to(device)
        seg_end   = self.segment_end_transform.to(device)
        segment_start = seg_start @ vertices
        segment_end   = seg_end   @ vertices
        return segment_start, segment_end

    def __call__(self, ego_state, ego_extent, adv_state, adv_extent):
        """Compute polygon collision costs on a single, consistent device."""
        # pick device from positions
        device = ego_state['pos'].device
        ego_pos = ego_state["pos"].to(device)
        ego_yaw = ego_state["yaw"].to(device)
        ego_extent = torch.diag_embed(ego_extent).to(device)

        adv_pos = adv_state["pos"].to(device)
        adv_yaw = adv_state["yaw"].to(device)
        adv_extent = torch.diag_embed(adv_extent).to(device)

        pos = torch.cat([ego_pos, adv_pos], dim=1)
        yaw = torch.cat([ego_yaw, adv_yaw], dim=1)
        extent = torch.cat([ego_extent, adv_extent], dim=1)

        rot_mat = torch.cat([
            torch.cos(yaw), -torch.sin(yaw),
            torch.sin(yaw),  torch.cos(yaw),
        ], dim=-1).view(self.batch_size, self.num_agents+1, 2, 2).to(device)

        corners = self.unit_square.to(device).expand(self.batch_size, self.num_agents+1, 4, 2) @ extent

        corners = corners @ rot_mat.permute(0, 1, 3, 2)

        corners = corners + pos.unsqueeze(-2)

        segment_starts, segment_ends = self.vertices_to_edges_vectorized(corners)
        segments = segment_ends - segment_starts

        corners = corners.repeat_interleave(self.num_agents+1, dim=1)
        segment_starts = segment_starts.repeat(1, self.num_agents+1, 1, 1)
        segment_ends = segment_ends.repeat(1, self.num_agents+1, 1, 1)
        segments = segments.repeat(1, self.num_agents+1, 1, 1)

        corners = corners.repeat_interleave(4, dim=2)
        segment_starts = segment_starts.repeat(1, 1, 4, 1)
        segment_ends = segment_ends.repeat(1, 1, 4, 1)
        segments = segments.repeat(1, 1, 4, 1)

        projections = torch.matmul(
            (corners - segment_starts).unsqueeze(-2),
            segments.unsqueeze(-1)
        ).squeeze(-1)

        projections = projections / torch.sum(segments**2,dim=-1, keepdim=True)

        projections = torch.clamp(projections, 0., 1.)

        closest_points = segment_starts + segments * projections

        distances = torch.norm(corners - closest_points, dim=-1, keepdim=True)
        closest_points_list = closest_points.view(-1,2).clone()

        distances, distances_idxs = torch.min(distances, dim=-2)

        distances_idxs = distances_idxs.unsqueeze(-1).repeat(1, 1, 1, 2)

        distances = distances.view(self.batch_size, self.num_agents + 1, self.num_agents + 1, 1)

        n = self.num_agents + 1
        distances = distances[0, :, :, 0].flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)

        ego_cost = torch.min(distances[0])[None, None]

        if distances.size(0) > 2:
            distances_adv = distances[1:, 1:]
            adv_cost = torch.min(distances_adv, dim=-1)[0][None]
        else:
            adv_cost = torch.zeros(1, 0, device=device)

        return ego_cost, adv_cost, closest_points_list
