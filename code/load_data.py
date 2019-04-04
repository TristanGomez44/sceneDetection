
import shotDetect
import subprocess
import xml.etree.ElementTree as ET
import os
import sys
import cv2
import numpy as np
import glob
import modelBuilder
import torch
def getMiddleFrames(dataset):
    #Get the middles frames of each shot for all video in a dataset
    #Require the shot to be pre-computed.

    videoPathLists = glob.glob("../data/{}/*.*".format(dataset))

    for videoPath in videoPathLists:

        dirname = "../data/{}/{}/".format(dataset,os.path.splitext(os.path.basename(videoPath))[0])
        if not os.path.exists(dirname+"/middleFrames/"):
            print(videoPath)

            videoName = os.path.splitext(os.path.basename(videoPath))[0]

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

            if dataset == "OVSD":
                sceneBounds = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,videoName),delimiter='\t')
            else:
                raise ValueError("Unknown dataset :".format(dataset))

            print("sceneBounds")
            print(sceneBounds)

            cap = cv2.VideoCapture(videoPath)
            success = True
            i = 0
            scene=0
            newSceneShotIndexs = []
            keyFrameInd = np.array(keyFrameInd)
            while success:
                success, imageRaw = cap.read()

                if i in keyFrameInd:
                    if scene < len(sceneBounds):
                        if sceneBounds[scene,1] < i:
                            print("Scene ",scene,"shot",np.where(keyFrameInd == i)[0][0],"frame",i,"time",(i//24)//60,"m",(i//24)%60,"s")
                            newSceneShotIndexs.append(np.where(keyFrameInd == i)[0][0])
                            scene += 1

                    cv2.imwrite(dirname+"/middleFrames/frame"+str(i)+".png",imageRaw)

                i += 1

            #This binary mask indicates if a shot is the begining of a new scene or not
            sceneTransition = np.zeros((len(keyFrameInd)))

            sceneTransition[newSceneShotIndexs] = 1

            print(sceneTransition.nonzero())

            np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,videoName),sceneTransition)

class SeqLoader():

    def __init__(self,dataset,batchSize,lMin,lMax,seed,imgSize,propStart,propEnd):

        self.batchSize = batchSize
        self.lMin = lMin
        self.lMax = lMax
        self.imgSize = imgSize
        self.dataset = dataset
        np.random.seed(seed)

        self.videoPathLists = sorted(glob.glob("../data/{}/*.*".format(dataset)))

        nbVid = len(self.videoPathLists)
        np.random.shuffle(self.videoPathLists)

        self.videoPathLists = self.videoPathLists[int(nbVid*propStart):int(nbVid*propEnd)]

        self.videoPathLists = self.videoPathLists

        self.framesDict = {}
        self.targetDict = {}

    def __iter__(self):

        seqList = []
        for videoPath in self.videoPathLists:

            vidName = os.path.splitext(os.path.basename(videoPath))[0]

            if not vidName in self.framesDict:
                self.framesDict[vidName]= np.array(sorted(glob.glob(os.path.splitext(videoPath)[0]+"/middleFrames/frame*.png"),key=modelBuilder.findNumbers))

            print(os.path.splitext(videoPath)[0]+"/frame*.png")

            frameInd = 0
            frameNb = len(self.framesDict[vidName])

            seqLengths = np.random.randint(self.lMin,self.lMax,size=(frameNb//self.lMin)+1)

            seqInds = np.cumsum(seqLengths)

            #The number of the last sequence made with the video
            lastSeqNb = np.where((seqInds >= frameNb-self.lMin))[0][0]

            seqInds[lastSeqNb] = frameNb

            seqInds = seqInds[:lastSeqNb+1]

            starts = seqInds[:-1]
            ends = seqInds[1:]
            names = np.array(vidName)[np.newaxis].repeat(len(seqInds)-1)

            seqList.extend([{"vidName":vidName,"start":start,"end":end} for vidName,start,end in zip(names,starts,ends)])

            self.targetDict[vidName] = torch.tensor(np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(self.dataset,vidName)))

        self.seqList = np.array(seqList,dtype=object)
        np.random.shuffle(self.seqList)
        self.currInd = 0

        return self

    def __next__(self):

        if self.currInd > len(self.seqList):
            raise StopIteration

        batchSize = min(self.batchSize,len(self.seqList)-self.currInd)

        batchTensor = torch.zeros((batchSize,self.lMax,self.imgSize[0],self.imgSize[1],3))
        targetTensor = torch.zeros((batchSize,self.lMax))
        seqLenTensor = torch.zeros((batchSize))
        i=0

        seqList = self.seqList[self.currInd:self.currInd+batchSize]
        print("Loading batch")
        for i,seq in enumerate(seqList):
            print("\t",i,"/",len(seqList))
            inTensor = torch.tensor(list(map(lambda x:cv2.resize(cv2.imread(x), self.imgSize),self.framesDict[seq["vidName"]][seq["start"]:seq["end"]+1])))

        self.currInd += self.batchSize
        sys.exit(0)
        return batchTensor,targetTensor,seqLenTensor

def main():

    loader = SeqLoader("OVSD",4,10,20,1,(100,100))
    loader.initLoader()
    print("Get batch")
    x,y,lens = loader.getBatch()
    print(x.sum(),y,lens)
if __name__ == "__main__":
    main()
