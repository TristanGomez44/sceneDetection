
import shotDetect
import subprocess
import xml.etree.ElementTree as ET
import os
import sys
import cv2
import numpy as np
import glob

def getMiddleFrames(dataset):
    #Get the middles frames of each shot for all video in a dataset
    #Require the shot to be pre-computed.

    videoPathLists = glob.glob("../data/{}/*.*".format(dataset))

    for videoPath in videoPathLists:

        #Middle frame extraction
        if not os.path.exists(dirname+"/middleFrames/"):
            print(videoPath)

            dirname = "../data/{}/{}/".format(dataset,os.path.splitext(os.path.basename(videoPath))[0])
            os.makedirs(dirname+"/middleFrames/")

            #Middle frame index extration
            print("\t Middle frame extraction")
            tree = ET.parse("{}/result.xml".format(dirname)).getroot()
            shots = tree.find("content").find("body").find("shots")

            frameNb = int(shots[-1].get("fduration"))+int(shots[-1].get("fbegin"))

            bounds = list(map(lambda x:int(x.get("fbegin")),shots))
            bounds.append(frameNb)
            keyFrameInd = []

            for i,shot in enumerate(bounds):
                if i < len(bounds)-1:
                    keyFrameInd.append((bounds[i]+bounds[i+1])//2)

            cap = cv2.VideoCapture(videoPath)
            success = True
            i = 0
            while success:
                success, imageRaw = cap.read()

                if i in keyFrameInd:
                    cv2.imwrite(dirname+"/middleFrames/frame"+str(i)+".png",imageRaw)

                i += 1

if __name__ == "__main__":
    main()
