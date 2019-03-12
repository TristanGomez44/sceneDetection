
import shotDetect
import subprocess
import xml.etree.ElementTree as ET
import os
import sys
import cv2
def main():
    print("Loading data")

    videoPath = "../data/big_buck_bunny_480p_surround-fix.avi"

    if not os.path.exists("../data/result.xml"):

        output = subprocess.Popen(("shotdetect",
                            "-i",
                            videoPath,
                            "-s",
                            str(60),
                            "-o",
                            os.path.dirname(videoPath),
                            "-f","-l","-m"),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

        output = output.stdout.read()


    tree = ET.parse("../data/result.xml").getroot()
    shots = tree.find("content").find("body").find("shots")

    frameNb = int(shots[-1].get("fduration"))+int(shots[-1].get("fbegin"))

    bounds = list(map(lambda x:int(x.get("fbegin")),shots))
    bounds.append(frameNb)
    keyFrameInd = []

    for i,shot in enumerate(bounds):
        if i < len(bounds)-1:
            keyFrameInd.append((bounds[i]+bounds[i+1])//2)

    #Writing the middle frame of each shot in a folder
    dirname = "../data/middleFrames/{}".format(os.path.basename(videoPath).replace(".avi",""))
    if not os.path.exists(dirname):

        os.makedirs(dirname)

        cap = cv2.VideoCapture(videoPath)
        success = True
        i = 0
        while success:
            success, imageRaw = cap.read()

            if i in keyFrameInd:
                cv2.imwrite(dirname+"/frame"+str(i)+".png",imageRaw)

            i += 1


if __name__ == "__main__":
    main()
