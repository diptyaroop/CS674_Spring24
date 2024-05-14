_base_ = '../default.py'

expname = 'dvgo_bottle'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/bottle',
    dataset_type='blender',
    white_bkgd=True,
)
