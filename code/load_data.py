
import shotDetect
import subprocess
import xml.etree.ElementTree as ET
import os
import sys
import cv2
import numpy as np
def main():

    datasetPath = "../data/test.csv"

    videoPaths = np.genfromtxt(datasetPath,dtype=str)
    #print(videoPaths)
    for videoPath in videoPaths:

        print(videoPath)

        dirname = "../data/{}/".format(os.path.splitext(os.path.basename(videoPath))[0])

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        #Shot extraction
        if not os.path.exists("{}/result.xml".format(dirname)):

            print("\t Shot detection")
            output = subprocess.Popen(("shotdetect",
                                "-i",
                                videoPath,
                                "-s",
                                str(60),
                                "-o",
                                dirname,
                                "-f","-l","-m"),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

            output = output.stdout.read()

        #Middle frame index extration
        print("\t Middle frame extraction")
        tree = ET.parse("../data/{}/result.xml".format(dirname)).getroot()
        shots = tree.find("content").find("body").find("shots")

        frameNb = int(shots[-1].get("fduration"))+int(shots[-1].get("fbegin"))

        bounds = list(map(lambda x:int(x.get("fbegin")),shots))
        bounds.append(frameNb)
        keyFrameInd = []

        for i,shot in enumerate(bounds):
            if i < len(bounds)-1:
                keyFrameInd.append((bounds[i]+bounds[i+1])//2)

        #Middle frame extraction
        if not os.path.exists(dirname+"/middleFrames/"):

            os.makedirs(dirname+"/middleFrames/")

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
