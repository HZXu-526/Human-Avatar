from third_parties.pytorch3d import ops
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

class SMPLDeformer():
    def __init__(self, model_path, gender, threshold=0.05, k=1) -> None:
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

        self.k = k
        self.threshold = threshold
        self.strategy = "nearest_neighbor"

        # define template canonical pose
        #   T-pose does't work very well for some cases (legs are too close) 
        self.initialized = False

    def initialize(self, betas, device):
        # convert to canonical space: deformed => T pose => Template pose
        batch_size = betas.shape[0]
        #print(batch_size,betas.shape)
        body_pose_t = torch.zeros((batch_size, 63), device=device)
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6



        smpl_outputs = self.body_model(left_hand_pose=None,
                                 right_hand_pose=None,
                                 jaw_pose=None,
                                 leye_pose=None,
                                 reye_pose=None,
                                 return_full_pose=True,
                                 expression=None,
                                 betas=betas, body_pose=body_pose_t)
        self.bbox = get_bbox_from_smpl(smpl_outputs.vertices[0:1].detach())

        self.T_template = smpl_outputs.T
        self.vs_template = smpl_outputs.vertices
        self.pose_offset_t = smpl_outputs.pose_offsets
        self.shape_offset_t = smpl_outputs.shape_offsets

    def get_bbox_deformed(self):
        return get_bbox_from_smpl(self.vertices[0:1].detach())

    def prepare_deformer(self, smpl_params):
        device = smpl_params["betas"].device
        if next(self.body_model.parameters()).device != device:
            self.body_model = self.body_model.to(device)
            # self.body_model.eval()

        if not self.initialized:
            self.initialize(smpl_params["betas"], device)

            # TODO: we have to intialized it every time because betas might change
            # self.initialized = True

        smpl_outputs = self.body_model(left_hand_pose=None,
                                        right_hand_pose=None,
                                        jaw_pose=None,
                                        leye_pose=None,
                                        reye_pose=None,
                                        return_full_pose=True,
                                        expression=None,
                                       betas=smpl_params["betas"],
                                       body_pose=smpl_params["body_pose"],
                                       global_orient=smpl_params["global_orient"],
                                       transl=smpl_params["transl"])

        # remove & reapply the blendshape
        s2w = smpl_outputs.A[:, 0]
        w2s = torch.inverse(s2w)
        T_inv = torch.inverse(smpl_outputs.T.float()).clone() @ s2w[:, None]
        T_inv[..., :3, 3] += self.pose_offset_t - smpl_outputs.pose_offsets
        T_inv[..., :3, 3] += self.shape_offset_t - smpl_outputs.shape_offsets
        T_inv = self.T_template @ T_inv
        self.T_inv = T_inv
        self.vertices = (smpl_outputs.vertices @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        self.w2s = w2s

    def transform_rays_w2s(self, rays):
        """transform rays from world to smpl coordinate system"""
        w2s = self.w2s
        rays.o = (rays.o @ w2s[:, :3, :3].permute(0, 2, 1)) + w2s[:, None, :3, 3]
        rays.d = (rays.d @ w2s[:, :3, :3].permute(0, 2, 1)).to(rays.d)
        d = torch.norm(rays.o, dim=-1)
        rays.near = d - 1
        rays.far = d + 1

    def deform(self, pts):
        """transform pts to canonical space"""
        # construct template pose
        batch_size = self.vertices.shape[0]

        # find nearest neighbors
        pts = pts.reshape(batch_size, -1, 3)
        with torch.no_grad():
            dist_sq, idx,_ = ops.knn_points(pts.float(), self.vertices.float(), K=self.k)

        # if valid => return the transformed points
        valid = dist_sq < self.threshold ** 2

        # squeeze because we use the nearest neighbor only
        idx = idx.squeeze(-1)
        valid = valid.squeeze(-1)

        # T ~ (batch_size, #pts, #neighbors, 4, 4)
        pts_cano = torch.zeros_like(pts, dtype=torch.float32)
        for i in range(batch_size):
            # mask = valid[i]
            Tv_inv = self.T_inv[i][idx[i]]
            pts_cano[i] = (Tv_inv[..., :3, :3] @ pts[i][..., None]).squeeze(-1) + Tv_inv[..., :3, 3]
        return pts_cano.reshape(-1, 3), valid.reshape(-1)

    def deform_train(self, pts, model):
        pts_cano, valid = self.deform(pts)
        rgb_cano = torch.zeros_like(pts, dtype=torch.float32)
        sigma_cano = torch.ones_like(pts[..., 0]) * -1e5
        if valid.any():
            with torch.cuda.amp.autocast():
                rgb_cano[valid], sigma_cano[valid] = model(pts_cano[valid], None)
            valid = torch.isfinite(rgb_cano).all(-1) & torch.isfinite(sigma_cano)
            rgb_cano[~valid] = 0
            sigma_cano[~valid] = -1e5
        return rgb_cano, sigma_cano

    def deform_test(self, pts, model):
        pts_cano, valid = self.deform(pts)
        rgb_cano = torch.zeros_like(pts, dtype=torch.float32)
        sigma_cano = torch.zeros_like(pts[..., 0])
        if valid.any():
            with torch.cuda.amp.autocast():
                rgb_cano[valid], sigma_cano[valid] = model(pts_cano[valid], None)
        return rgb_cano, sigma_cano

    def __call__(self, pts, model, eval_mode=True):
        if eval_mode:
            return self.deform_test(pts, model)
        else:
            return self.deform_train(pts, model)
