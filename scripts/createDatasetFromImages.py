
# Script to create train/val/test dataset from images generated using colmap2NeRF.py
# Check this link for more imfo on generating images: https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md

import sys
import os
import shutil
from pathlib import Path
import json

# Splits images into train/val/test
# Considers 400 images (100 - train, 100 - val, 200 - test). SImilar to NeRF synthetic dataset.
# Remaining images (if any) are discarded. Needs at least 400 images to run correctly
def splitImages(inputPath, outputPath):
    
    # Images are named as 0<x>.png, where x is from 001 to 399
    # x%4 == 1 --> train image, x%4 == 2 --> val image, x%4 == 3 or 0 --> test images

    trainOutputPath = outputPath+"/"+"train"
    valOutputPath = outputPath+"/"+"val"
    testOutputPath = outputPath+"/"+"test"

    trainIdx = 0
    valIdx = 0
    testIdx = 0

    for x in range(1, 401):
        file = inputPath+"/"+"0"+str(x).zfill(3)+".png"
        print(file)
        # if (x%4 == 1): # train image
        if (x<=100):
            print("train image")
            Path(trainOutputPath).mkdir(parents=True, exist_ok=True)
            # shutil.copy(file, trainOutputPath+"/r_"+str(trainIdx)+".png")
            shutil.copy(file, trainOutputPath+"/r_"+str(x-1)+".png")
            trainIdx += 1
        # elif (x%4 == 2): # val image
        elif (x<=200):
            print("val image")
            Path(valOutputPath).mkdir(parents=True, exist_ok=True)
            # shutil.copy(file, valOutputPath+"/r_"+str(valIdx)+".png")
            shutil.copy(file, valOutputPath+"/r_"+str(x-101)+".png")
            valIdx += 1
        else: # test image
            print("test image")
            Path(testOutputPath).mkdir(parents=True, exist_ok=True)
            # shutil.copy(file, testOutputPath+"/r_"+str(testIdx)+".png")
            shutil.copy(file, testOutputPath+"/r_"+str(x-201)+".png")
            testIdx += 1
    return

# Splits transforms.json into transforms_<train/val/test>.json for the corresponding images
def splitTransformsFile(outputPath, numTrainFrames=100, numValFrames=100, numTestFrames=200, totalFrames=400):
    transformsFile = outputPath+"/transforms.json"

    with open(transformsFile) as jsonFile:
        transforms = json.load(jsonFile)
    transformsTrain = transforms.copy()
    transformsVal = transforms.copy()
    transformsTest = transforms.copy()

    trainFrames = [None] * numTrainFrames
    valFrames = [None] * numValFrames
    testFrames = [None] * numTestFrames

    for frame in transforms["frames"]:
        filePath = frame["file_path"]
        filePath = filePath.split("/")[-1]
        print(filePath)
        fileName = filePath.split(".")[0]
        if (int(fileName)>totalFrames):
            print("Excluding frames > ", totalFrames)
            continue
        # if (int(fileName)%4 == 1): # train frame
        if (int(fileName) <= 100):
            trainFrame = frame.copy()
            trainFrameNum = int(fileName)//4
            # filePath = "./train/r_"+str(trainFrameNum)
            filePath = "./train/r_"+str(int(fileName)-1)
            trainFrame["file_path"] = filePath
            print(trainFrame["file_path"])
            # trainFrames[trainFrameNum] = trainFrame
            trainFrames[int(fileName)-1] = trainFrame
        # elif (int(fileName)%4 == 2): # val frame
        elif (int(fileName) <= 200):
            valFrame = frame.copy()
            valFrameNum = int(fileName)//4
            # filePath = "./val/r_"+str(valFrameNum)
            filePath = "./val/r_"+str(int(fileName)-101)
            valFrame["file_path"] = filePath
            print(valFrame["file_path"])
            # valFrames[valFrameNum] = valFrame
            valFrames[int(fileName)-101] = valFrame
        else:
            testFrame = frame.copy()
            testFrameNum = int(fileName)//2-1
            # filePath = "./test/r_"+str(testFrameNum)
            filePath = "./test/r_"+str(int(fileName)-201)
            testFrame["file_path"] = filePath
            print(testFrame["file_path"])
            # testFrames[testFrameNum] = testFrame
            testFrames[int(fileName)-201] = testFrame

    transformsTrain["frames"] = trainFrames
    transformsVal["frames"] = valFrames
    transformsTest["frames"] = testFrames

    with open(outputPath+"/transforms_train.json", "w") as outfile:
        data = json.dumps(transformsTrain, indent=4)
        print(data , file=outfile)

    with open(outputPath+"/transforms_val.json", "w") as outfile: 
        data = json.dumps(transformsVal, indent=4)
        print(data , file=outfile)

    with open(outputPath+"/transforms_test.json", "w") as outfile: 
        data = json.dumps(transformsTest, indent=4)
        print(data , file=outfile)

    return


if __name__ == "__main__":
    if (len(sys.argv)<2):
        print("Usage: python3 createDatasetFromImages.py <dataset_name>")
        print("Dataset name eg.: bottle")
        exit(0)

    dataset = sys.argv[1]
    inputPath = f"../data/nerf_synthetic/{dataset}/clean_images"
    outputPath = f"../data/nerf_synthetic/{dataset}"
    
    # splitImages(inputPath, outputPath)
    splitTransformsFile(outputPath)