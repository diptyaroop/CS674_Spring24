import numpy as np

from .load_blender import load_blender_data


def load_data(args):

    K, depths = None, None
    near_clip = None

    
    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near, far = 2., 6.

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
            else:
                images = images[...,:3]*images[...,-1:]
    else:
        raise NotImplementedError(f'Unknown dataset type {args.dataset_type} exiting')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_train=i_train, i_val=i_val, i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,
        irregular_shape=irregular_shape,
    )
    return data_dict
