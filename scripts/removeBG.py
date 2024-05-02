# Script to remove background from dataset images for NeRF. Original images are generated using colmap2NeRF.py.
# Check this link for more imfo on generating images: https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md
import os
import sys
from pathlib import Path
from rembg import remove 
from PIL import Image 


def removeBackgroundFromImages(inputPath, outputPath):

    fileList = os.listdir(inputPath)
    sortedFiles = sorted(fileList)
    for file in sortedFiles:
        if (int(file.split(".")[0])<203):
            continue
        print("Image: ", file)

        # Processing the image. Image name format: 0<x>.jpg (obtained from colmap2Nerf.py)
        input = Image.open(inputPath+"/"+file) 
        # Removing the background from the given Image 
        output = remove(input) 
        # Creating output path if it doesn't exist
        Path(outputPath).mkdir(parents=True, exist_ok=True)
        #Saving the image in the given path. Output image name format: 0<x>.png
        output.save(outputPath+"/"+file.split(".")[0]+".png") 

    return


if __name__ == "__main__":
    if (len(sys.argv)<2):
        print("Usage: python3 removeBG.py <dataset_name>")
        print("Dataset name eg.: bottle")
        exit(0)

    dataset = sys.argv[1]
    inputPath = f"../data/nerf_synthetic/{dataset}/images"
    outputPath = f"../data/nerf_synthetic/{dataset}/clean_images" 
    
    removeBackgroundFromImages(inputPath, outputPath)