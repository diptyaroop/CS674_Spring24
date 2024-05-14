_base_ = '../default.py'

expname = 'dvgo_lego_sty'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego_sty',
    dataset_type='blender',
    white_bkgd=True,
)

