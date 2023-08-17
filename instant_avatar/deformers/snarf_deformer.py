from .fast_snarf.deformer_torch import ForwardDeformer
from .smplx import SMPL,SMPLH
from smplx.utils import Struct
import torch
import hydra
import numpy as np

vertex_ids = {
    'smplh': {
        'nose':		    332,
        'reye':		    6260,
        'leye':		    2800,
        'rear':		    4071,
        'lear':		    583,
        'rthumb':		6191,
        'rindex':		5782,
        'rmiddle':		5905,
        'rring':		6016,
        'rpinky':		6133,
        'lthumb':		2746,
        'lindex':		2319,
        'lmiddle':		2445,
        'lring':		2556,
        'lpinky':		2673,
        'LBigToe':		3216,
        'LSmallToe':	3226,
        'LHeel':		3387,
        'RBigToe':		6617,
        'RSmallToe':    6624,
        'RHeel':		6787
    },
    'smplx': {
        'nose':		    9120,
        'reye':		    9929,
        'leye':		    9448,
        'rear':		    616,
        'lear':		    6,
        'rthumb':		8079,
        'rindex':		7669,
        'rmiddle':		7794,
        'rring':		7905,
        'rpinky':		8022,
        'lthumb':		5361,
        'lindex':		4933,
        'lmiddle':		5058,
        'lring':		5169,
        'lpinky':		5286,
        'LBigToe':		5770,
        'LSmallToe':    5780,
        'LHeel':		8846,
        'RBigToe':		8463,
        'RSmallToe': 	8474,
        'RHeel':  		8635
    },
    'mano': {
            'thumb':		744,
            'index':		320,
            'middle':		443,
            'ring':		    554,
            'pinky':		671,
        }
}

def get_predefined_rest_pose(cano_pose, device="cuda"):
    body_pose_t = torch.zeros((1, 63), device=device)
    if cano_pose.lower() == "da_pose":
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
    elif cano_pose.lower() == "a_pose":
        body_pose_t[:, 2] = 0.2
        body_pose_t[:, 5] = -0.2
        body_pose_t[:, 47] = -0.8
        body_pose_t[:, 50] = 0.8
    else:
        raise ValueError("Unknown cano_pose: {}".format(cano_pose))
    return body_pose_t

def get_bbox_from_smpl(vs, factor=1.2):
    assert vs.shape[0] == 1
    min_vert = vs.min(dim=1).values
    max_vert = vs.max(dim=1).values

    c = (max_vert + min_vert) / 2
    s = (max_vert - min_vert) / 2
    s = s.max(dim=-1).values * factor

    min_vert = c - s[:, None]
    max_vert = c + s[:, None]
    return torch.cat([min_vert, max_vert], dim=0)

class SNARFDeformer():
    def __init__(self, model_path, gender, opt) -> None:
        model_path = hydra.utils.to_absolute_path(model_path)
        bm_path = "/home/wjx/Desktop/money/InstantAvatar/data/SMPLX/smpl/neutral/model.npz"
        data_struct = None
        if '.npz' in bm_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            data_struct.hands_componentsl = np.zeros((0))
            data_struct.hands_componentsr = np.zeros((0))
            data_struct.hands_meanl = np.zeros((15 * 3))
            data_struct.hands_meanr = np.zeros((15 * 3))
            V, D, B = data_struct.shapedirs.shape
            data_struct.shapedirs = np.concatenate([data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM - B))],
                                                   axis=-1)  # super hacky way to let smplh use 16-size beta
            kwargs = {
                'model_type': 'smplh',
                'data_struct': data_struct,
                'num_betas': 16,
                'batch_size': 1,
                'num_expression_coeffs': 10,
                'vertex_ids': vertex_ids['smplh'],
                'use_pca': False,
                'flat_hand_mean': True
            }

        self.body_model = SMPLH(bm_path, **kwargs)
        self.deformer = ForwardDeformer(opt)
        self.initialized = False
        self.opt = opt

    def initialize(self, betas, device):
        if isinstance(self.opt.cano_pose, str):
            body_pose_t = get_predefined_rest_pose(self.opt.cano_pose, device=device)
        else:
            body_pose_t = torch.zeros((1, 63), device=device)
            body_pose_t[:, 2] = self.opt.cano_pose[0]
            body_pose_t[:, 5] = self.opt.cano_pose[1]
            body_pose_t[:, 47] = self.opt.cano_pose[2]
            body_pose_t[:, 50] = self.opt.cano_pose[3]

        # print(body_pose_t.shape)
        # print(betas.shape)
        # betas = torch.squeeze(betas)
        smpl_outputs = self.body_model(left_hand_pose=None,
                                 right_hand_pose=None,
                                 jaw_pose=None,
                                 leye_pose=None,
                                 reye_pose=None,
                                 return_full_pose=True,
                                 expression=None,betas=betas, body_pose=body_pose_t)
        self.tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach())
        self.vs_template = smpl_outputs.vertices

        # initialize SNARF
        self.deformer.device = device
        self.deformer.switch_to_explicit(resolution=self.opt.resolution,
                                         smpl_verts=smpl_outputs.vertices.float().detach(),
                                         smpl_weights=self.body_model.lbs_weights.clone()[None].detach(),
                                         use_smpl=True)
        self.bbox = get_bbox_from_smpl(smpl_outputs.vertices.detach())

        self.dtype = torch.float32
        self.deformer.lbs_voxel_final = self.deformer.lbs_voxel_final.type(self.dtype)
        self.deformer.grid_denorm = self.deformer.grid_denorm.type(self.dtype)
        self.deformer.scale = self.deformer.scale.type(self.dtype)
        self.deformer.offset = self.deformer.offset.type(self.dtype)
        self.deformer.scale_kernel = self.deformer.scale_kernel.type(self.dtype)
        self.deformer.offset_kernel = self.deformer.offset_kernel.type(self.dtype)

    def prepare_deformer(self, smpl_params):
        # for k in smpl_params:
        #     print(k,smpl_params[k].shape)
        device = smpl_params["betas"].device
        if next(self.body_model.parameters()).device != device:
            self.body_model = self.body_model.to(device)
        if not self.initialized:
            self.initialize(smpl_params["betas"], smpl_params["betas"].device)
            self.initialized = True

        smpl_outputs = self.body_model(pose_hand=None,
                                       betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"])
        s2w = smpl_outputs.A[:, 0].float()
        w2s = torch.inverse(s2w)

        tfs = (w2s[:, None] @ smpl_outputs.A.float() @ self.tfs_inv_t).type(self.dtype)
        self.deformer.precompute(tfs)

        self.w2s = w2s
        self.vertices = (smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        self.tfs = tfs
        self.smpl_outputs = smpl_outputs
        self.smpl_params = smpl_params

    def transform_rays_w2s(self, rays):
        """transform rays from world to smpl coordinate system"""
        w2s = self.w2s

        rays.o = (rays.o @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        rays.d = (rays.d @ w2s[:, :3, :3].permute(0, 2, 1)).to(rays.d)
        d = torch.norm(rays.o, dim=-1)
        rays.near = d - 1
        rays.far = d + 1

    def get_bbox_deformed(self):
        voxel = self.deformer.voxel_d[0].reshape(3, -1)
        return [voxel.min(dim=1).values, voxel.max(dim=1).values]

    def deform(self, pts, eval_mode):
        """transform pts to canonical space"""
        point_size = pts.shape[0]
        betas = self.smpl_outputs.betas
        batch_size = betas.shape[0]
        pts = pts.reshape(batch_size, -1, 3)

        pts_cano, others = self.deformer.forward(pts, cond=None, tfs=self.tfs, eval_mode=eval_mode)
        valid = others["valid_ids"].reshape(point_size, -1)

        # from pytorch3d import ops
        # with torch.no_grad():
        #     dist_sq, idx, neighbors = ops.knn_points(pts_cano.reshape(1, -1, 3), self.vs_template, K=1)
        #     dist_sq = dist_sq.reshape(valid.shape)
        # valid = (dist_sq < 0.05 ** 2) & valid
        return pts_cano.reshape(point_size, -1, 3), valid

    @torch.no_grad()
    def deform_test(self, pts, model):
        pts_cano_all, valid = self.deform(pts.type(self.dtype), eval_mode=True)

        rgb_cano = torch.zeros_like(pts_cano_all).float()
        sigma_cano = torch.zeros_like(pts_cano_all[..., 0]).float()
        if valid.any():
            with torch.cuda.amp.autocast():
                rgb_cano[valid], sigma_cano[valid] = model(pts_cano_all[valid], None)

            sigma_cano[valid] = torch.nan_to_num(sigma_cano[valid], 0, 0, 0)
            rgb_cano[valid] = torch.nan_to_num(rgb_cano[valid], 0, 0, 0)

        sigma_cano, idx = torch.max(sigma_cano, dim=-1)
        rgb_cano = torch.gather(rgb_cano, 1, idx[:, None, None].repeat(1, 1, 3))
        return rgb_cano.reshape(-1, 3), sigma_cano.reshape(-1)

    def deform_train(self, pts, model):
        pts_cano_all, valid = self.deform(pts.type(self.dtype), eval_mode=False)

        rgb_cano = torch.zeros_like(pts_cano_all).float()
        sigma_cano = -torch.ones_like(pts_cano_all[..., 0]).float() * 1e5

        if valid.any():
            if not torch.isfinite(pts_cano_all).all():
                print("WARNING: NaN found in pts_cano_all")
            with torch.cuda.amp.autocast():
                rgb_cano[valid], sigma_cano[valid] = model(pts_cano_all[valid], None)
            if not torch.isfinite(sigma_cano).all():
                print("WARNING: NaN found in sigma_cano")

        sigma_cano, idx = torch.max(sigma_cano, dim=-1)
        rgb_cano = torch.gather(rgb_cano, 1, idx[:, None, None].repeat(1, 1, 3))
        return rgb_cano.reshape(-1, 3), sigma_cano.reshape(-1)

    def __call__(self, pts, model, eval_mode=True):
        if eval_mode:
            return self.deform_test(pts, model)
        else:
            return self.deform_train(pts, model)
