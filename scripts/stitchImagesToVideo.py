import os
import sys
import imageio.v2 as imageio

def createVideoFromImages(inputPath):

    files = []
    for file in os.listdir(inputPath):
        if ("png" in file):
            files.append(inputPath+file)
    files = sorted(files)
    # print(files)

    writer = imageio.get_writer(inputPath+'lego_nerf_sty_video.mp4', fps=20)

    for im in files:
        writer.append_data(imageio.imread(im))
    writer.close()
    return



if __name__ =="__main__":
    
    if (len(sys.argv)<2):
        print("Usage: python3 stitchImagesToVideo.py <input_path>")
        exit(0)
    
    inputPath = sys.argv[1]

    createVideoFromImages(inputPath)