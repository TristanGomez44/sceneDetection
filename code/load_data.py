
import shotDetect
import subprocess
import xml.etree.ElementTree as ET
import os
import sys
import cv2
import numpy as np
import glob
import modelBuilder
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

class SeqLoader():

    def __init__(self,dataset,batchSize,lMin,lMax,seed,imgSize):

        self.batchSize = batchSize
        self.lMin = lMin
        self.lMax = lMax
        self.imgSize = imgSize
        self.dataset = dataset
        np.random.seed(seed)

        videoPathLists = glob.glob("../data/{}/*.*".format(dataset))

        seqList = []
        self.framesDict = {}
        for videoPath in videoPathLists:

            middleFramePathList = np.array(sorted(glob.glob(os.path.splitext(videoPath)[0]+"/middleFrames/frame*.png"),key=modelBuilder.findNumbers))
            print(os.path.splitext(videoPath)[0]+"/frame*.png")

            vidName = os.path.basename(videoPath)
            frameInd = 0
            frameNb = len(middleFramePathList)

            seqLengths = np.random.randint(lMin,lMax,size=(frameNb//lMin)+1)

            seqInds = np.cumsum(seqLengths)

            #The number of the last sequence made with the video
            lastSeqNb = np.where((seqInds >= frameNb-lMin))[0][0]

            seqInds[lastSeqNb] = frameNb

            seqInds = seqInds[:lastSeqNb+1]

            starts = seqInds[:-1]
            ends = seqInds[1:]
            names = np.array(vidName)[np.newaxis].repeat(len(seqInds)-1)

            seqList.extend([{"vidName":vidName,"start":start,"end":end} for vidName,start,end in zip(names,starts,ends)])
            self.framesDict[vidName] = middleFramePathList

        self.seqList = np.array(seqList,dtype=object)

        np.random.shuffle(self.seqList)

        self.currInd = 0

    def getBatch(self):

        if self.currInd > len(self.seqList):
            raise ValueError("No more batch to return")

        def readSeq(x):
            images = np.array(list(map(lambda x:cv2.resize(cv2.imread(x), self.imgSize),self.framesDict[x["vidName"]][x["start"]:x["end"]+1])))
            return images

        batch = np.array(list(map(readSeq,self.seqList[self.currInd:self.currInd+self.batchSize])))

        maxLen = max(list(map(lambda x:len(x),batch)))

        batchTensor = np.zeros((len(batch),maxLen,self.imgSize[0],self.imgSize[1],3))

        for i,seq in enumerate(batch):
            batchTensor[i,:len(batch[i])] = batch[i]

        self.currInd += self.batchSize

        return batch

def main():

    loader = SeqLoader("OVSD",32,10,20,1,(100,100))
    print("Get batch")
    loader.getBatch()

if __name__ == "__main__":
    main()
