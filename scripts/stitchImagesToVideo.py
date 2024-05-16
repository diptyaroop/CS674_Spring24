import os
import sys
import imageio.v2 as imageio

def createVideoFromImages(inputPath, dataset):

    files = []
    for file in os.listdir(inputPath):
        if ("png" in file):
            files.append(inputPath+"/"+file)
    files = sorted(files)
    # print(files)

    writer = imageio.get_writer(inputPath+"/"+dataset+'_nerf_sty_video.mp4', fps=20)

    for im in files:
        writer.append_data(imageio.imread(im))
    writer.close()
    return



if __name__ =="__main__":
    
    if (len(sys.argv)<3):
        print("Usage: python3 stitchImagesToVideo.py <input_path> <dataset>")
        exit(0)
    
    inputPath = sys.argv[1]
    dataset = sys.argv[2]

    createVideoFromImages(inputPath, dataset)