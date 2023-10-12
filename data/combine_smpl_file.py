import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract and modify pose parameters')
    parser.add_argument('--rate', type=int, default=1, help='Rate for extracting pose parameters')
    parser.add_argument('humor_params_path', help='Path to the humor results NPZ file')
    parser.add_argument('smpl_params_path', help='Path to the romp poses NPZ file')
    parser.add_argument('output_path', help='Path to save the output NPZ file')

    args = parser.parse_args()

    rate = args.rate
    humor_params = dict(np.load(args.humor_params_path))
    smpl_params = dict(np.load(args.smpl_params_path))

    extracted_humor_result = {"body_pose": [], "global_orient": []}
    for i in range(0, len(humor_params["pose_body"]), rate):
        extracted_humor_result["body_pose"].append(humor_params["pose_body"][i])
        extracted_humor_result["global_orient"].append(humor_params["root_orient"][i])

    smpl_params['body_pose'][:, :63] = extracted_humor_result['body_pose'][:]
    trans = 1 * smpl_params['transl']

    np.savez(args.output_path, **{
        "betas": smpl_params['betas'],
        "global_orient": extracted_humor_result["global_orient"],
        "body_pose": smpl_params['body_pose'],
        "transl": trans,
    })

if __name__ == '__main__':
    main()
