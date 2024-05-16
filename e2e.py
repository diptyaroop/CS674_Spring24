import os, sys, argparse
import signal

def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True,
                        help='the dataset to be used')
    parser.add_argument('--first_method', required=True,
                        help='the first method to be used. MUST BE \'nerf\' OR \'st\'')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true') # dump test images into testsavedir
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')
    parser.add_argument("--eval_coarse_grid", action="store_true") # [DM] render video after only training wrt coarse grid
    parser.add_argument("--stylized", action="store_true") # [DM] stylized_views

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    
    ######Style parser arguments
    #parser.add_argument('--content', default='./content', type=str,
    #                help='File path to the content image')
    #parser.add_argument('--content_dir', default='./content', type=str,
    #                    help='Directory path to a batch of content images')
    #parser.add_argument('--style', default='./style', type=str,
    #                    help='File path to the style image, or multiple style \
    #                    images separated by commas if you want to do style \
    #                    interpolation or spatial control')
    #parser.add_argument('--style_dir', default='./style', type=str,
    #                    help='Directory path to a batch of style images')
    #parser.add_argument('--output', type=str, default='output',
    #                    help='Directory to save the output image(s)')
    #parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
    #parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_25000.pth')
    #parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_25000.pth')
    #parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_25000.pth')


    parser.add_argument('--style_interpolation_weights', type=str, default="")
    parser.add_argument('--a', type=float, default=1.0)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
    return parser

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    #if (len(sys.argv)<3):
    #    print("Usage: python3 e2e.py <dataset_name> <nerf or sty> (optional)eval, where nerf or st is the method that is performed first")
    #    print("Dataset name eg.: bottle")
    #    exit(0)
    parser = config_parser()
    args = parser.parse_args()
    dataset = args.dataset
    inputPath = f"../data/nerf_synthetic/{dataset}/images"
    outputPath = None
    cwd = os.getcwd()

    bool_to_var_nerf = {
        "--no_reload" : args.no_reload,
        "--no_reload_optimizer" : args.no_reload_optimizer,
        "--render_only" : args.render_only,
        "--render_test" : args.render_test,
        "--render_train" : args.render_train,
        "--render_video" : args.render_video,
        "--render_video_flipy" : args.render_video_flipy,
        "--dump_images" : args.dump_images,
        "--eval_ssim" : args.eval_ssim,
        "--eval_lpips_alex" : args.eval_lpips_alex,
        "--eval_lpips_vgg" : args.eval_lpips_vgg,
        "--eval_coarse_grid" : args.eval_coarse_grid,
        "--stylized" : args.stylized
    }
    nerf_append_string = ""
    for arg_name, arg_value in bool_to_var_nerf.items():
        nerf_append_string += " " + arg_name if arg_value else ""
    firstMethod = args.first_method
    if firstMethod == "nerf":
        os.chdir(cwd+"/nerf")
        os.system(
        f"python3 run.py --config ./configs/nerf/{dataset}.py" + nerf_append_string
        )
        os.chdir('..')
        print("NeRF COMPONENT DONE")

        # Then run style transfer

        os.system("rm st/content/*")
        os.system(f"cp ./nerf/logs/nerf_synthetic/dvgo_{dataset}/hash_render_test_fine_last_new/*.png st/content/")
        os.chdir(cwd+"/st")
        os.system(
            f"python3 test.py --output ../outputs/{dataset}_nerf_sty "
            f" --a {args.a} "
            f"--position_embedding {args.position_embedding} --hidden_dim {args.hidden_dim}"
        )
        os.chdir('..')
        print("ST COMPONENT DONE")
        outputPath = f"./outputs/{dataset}_nerf_sty/"
        os.system(f"python3 scripts/stitchImagesToVideo.py {outputPath} {dataset}")
        print("VIDEO CREATED")
    elif firstMethod == "sty":
        # #run style transfer
        # os.chdir(cwd+"/st")
        # os.system(
        #     f"python3 test.py --output ../outputs/{dataset}_nerf_sty "
        #     f" --a {args.a} "
        #     f"--position_embedding {args.position_embedding} --hidden_dim {args.hidden_dim}"
        # )
        # os.chdir('..')
        # print("ST COMPONENT DONE")
       
        # os.chdir(cwd+"/nerf")
        # os.system(
        # f"python3 run.py --config ./configs/nerf/{dataset}.py" + nerf_append_string
        # )
        # os.chdir('..')
        # print("NeRF COMPONENT DONE")
        print("Due to time constraints we ewre not able to automate this. Please see the readme file for instructions on how to manually run this design.")
    else:
        print("Please choose either \"nerf\" or \"st\"")
        exit(0)
    
    # Stitching generated images into a video
   

    # os.system(f"python3 scripts/removeBG.py {inputPath} {outputPath}")
    
    # os.system(f"python3 scripts/createDatasetFromImages.py {inputPath} {outputPath}")
