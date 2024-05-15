import os, sys
import signal

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    if (len(sys.argv)<3):
        print("Usage: python3 e2e.py <dataset_name> <nerf or sty> (optional)eval, where nerf or st is the method that is performed first")
        print("Dataset name eg.: bottle")
        exit(0)

    dataset = sys.argv[1]
    inputPath = f"../data/nerf_synthetic/{dataset}/images"
    outputPath = None
    cwd = os.getcwd()
    print(cwd)
    
    firstMethod = sys.argv[2]
    if firstMethod == "nerf":
        # os.chdir(cwd+"/nerf")
        # os.system(f"python3 run.py --config ./configs/nerf/{dataset}.py --render_only --render_test --dump_images") # 
        # os.chdir('..')
        #Then run style transfer
        os.system(f"python3 st/test.py --content_dir ./nerf/logs/nerf_synthetic/{dataset} --output ./outputs/{dataset}_nerf_sty")
        #if sys.argv[3] == "eval":
        #    os.system("eval_metrics.py")
        # os.system(f"cp -r ./logs/nerf_synthetic/{dataset}_nerf_sty ./outputs/{dataset}_nerf_sty")
        outputPath = f"./outputs/{dataset}_nerf_sty"
    elif firstMethod == "sty":
        #run style transfer
        os.system(f"python3 st/test.py --contentdir ../data/nerf_synthetic/{dataset} --output ../data/nerf_synthetic/{dataset}_stylized")
        #if sys.argv[3] == "eval":
        #    os.system("eval_metrics.py")
        os.system(f"python3 nerf/run.py --config configs/nerf/{dataset}.py --render_test --dump_images --stylized")
        os.system(f"cp -r ./logs/nerf_synthetic/{dataset}_stylized ./outputs/{dataset}_sty_nerf")
    else:
        print("Please choose either \"nerf\" or \"st\"")
        exit(0)
    
    # Stitching generated images into a video
    os.system(f"python3 scripts/stitchImagesToVideo.py {outputPath}")

    # os.system(f"python3 scripts/removeBG.py {inputPath} {outputPath}")
    
    # os.system(f"python3 scripts/createDatasetFromImages.py {inputPath} {outputPath}")
