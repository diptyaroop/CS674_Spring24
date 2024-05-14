import os, sys
if __name__ == "__main__":
    if (len(sys.argv)<3):
        print("Usage: python3 e2e.py <dataset_name> <nerf or st> (optional)eval, where nerf or st is the method that is performed first")
        print("Dataset name eg.: bottle")
        exit(0)

    dataset = sys.argv[1]
    inputPath = f"../data/nerf_synthetic/{dataset}/images"
    outputPath = f"../data/nerf_synthetic/{dataset}/clean_images" 
    firstMethod = sys.argv[2]
    if firstMethod == "nerf":
        os.system(f"nerf/run.py --config configs/nerf/{dataset}.py --render_test --dump_images --i_print 500")
        #Then run style transfer
        os.system(f"ST/test.py --contentdir ../logs/nerf_synthetic/{dataset} --output ../logs/nerf_synthetic/{dataset}_final")
        #if sys.argv[3] == "eval":
        #    os.system("eval_metrics.py")
    elif firstMethod == "st":
        #run style transfer
        os.system(f"ST/test.py --contentdir ../data/nerf_synthetic/{dataset} --output ../data/nerf_synthetic/{dataset}_st")
        #if sys.argv[3] == "eval":
        #    os.system("eval_metrics.py")
        os.system(f"nerf/run.py --config configs/nerf/{dataset}.py --render_test --dump_images --i_print 500")
    else:
        print("Please choose either \"nerf\" or \"st\"")
        exit()
    os.system(f"nerf/removeBG.py {inputPath} {outputPath}")
    os.system(f"nerf/stitchImagesToVideo.py {inputPath}")
    os.system(f"nerf/createDatasetFromImages.py {inputPath} {outputPath}")
