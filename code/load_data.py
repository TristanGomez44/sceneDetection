
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

import subprocess

import soundfile as sf
import math
from PIL import Image
from torchvision import transforms

import vggish_input
def getMiddleFrames(dataset,audioLen=1):
    #Get the middles frames of each shot for all video in a dataset
    #Require the shot to be pre-computed.

    videoPathList = glob.glob("../data/{}/*.*".format(dataset))
    videoPathList = list(filter(lambda x:x.find(".wav")==-1,videoPathList))

    for videoPath in videoPathList:

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
            fps = float(tree.find("content").find("head").find("media").find("fps").text)

            #Extract the audio of the video
            audioPath = os.path.dirname(videoPath)+"/"+videoName+".wav"
            audioInfStr = tree.find("content").find("head").find("media").find("codec").find("audio").text

            audioSampleRate = int(audioInfStr.split(",")[1].replace(",","").replace("Hz","").replace(" ",""))
            audioBitRate = int(audioInfStr.split(",")[4].replace(",","").replace("kb/s","").replace(" ",""))*1000

            if not os.path.exists(audioPath):
                command = "ffmpeg -i {} -ab {} -ac 2 -ar {} -vn {}".format(videoPath,audioBitRate,audioSampleRate,audioPath)
                subprocess.call(command, shell=True)

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

            cap = cv2.VideoCapture(videoPath)
            success = True
            i = 0
            scene=0
            newSceneShotIndexs = []
            keyFrameInd = np.array(keyFrameInd)

            #Opening the sound file
            audioData, fs = sf.read(audioPath)
            #audioData= np.transpose(audioData,[1,0])

            while success:
                success, imageRaw = cap.read()
                if i in keyFrameInd:
                    if scene < len(sceneBounds):
                        if sceneBounds[scene,1] < i:
                            print("Scene ",scene,"shot",np.where(keyFrameInd == i)[0][0],"frame",i,"time",(i//fps)//60,"m",(i//fps)%60,"s")
                            newSceneShotIndexs.append(np.where(keyFrameInd == i)[0][0])
                            scene += 1

                    cv2.imwrite(dirname+"/middleFrames/frame"+str(i)+".png",imageRaw)

                    #Writing the audio sample
                    time = i/fps
                    pos = time*audioSampleRate
                    interv = audioLen*audioSampleRate/2
                    sampleToWrite = audioData[int(round(pos-interv)):int(round(pos+interv)),:]
                    fullArray = np.zeros((int(round(pos+interv))-int(round(pos-interv)),sampleToWrite.shape[1]))
                    fullArray[:len(sampleToWrite)] = sampleToWrite

                    sf.write(dirname+"/middleFrames/frame"+str(i)+".wav",fullArray,audioSampleRate,subtype='PCM_16',format="WAV")

                i += 1

            #This binary mask indicates if a shot is the begining of a new scene or not
            sceneTransition = np.zeros((len(keyFrameInd)))

            sceneTransition[newSceneShotIndexs] = 1

            print(sceneTransition.nonzero())

            np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,videoName),sceneTransition)

class SeqLoader():

    def __init__(self,dataset,batchSize,lMin,lMax,imgSize,propStart,propEnd,shuffle,removeUselessSeq=True,visualFeat=True,audioFeat=False,audioLen=-1):

        self.batchSize = batchSize
        self.lMin = lMin
        self.lMax = lMax
        self.imgSize = imgSize
        self.dataset = dataset
        self.shuffle = shuffle
        self.audioFeat = audioFeat
        self.visualFeat = visualFeat
        self.audioLen = audioLen
        self.videoPathLists = list(filter(lambda x:x.find(".wav") ==-1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))

        nbVid = len(self.videoPathLists)
        if shuffle:
            np.random.shuffle(self.videoPathLists)

        self.videoPathLists = self.videoPathLists[int(nbVid*propStart):int(nbVid*propEnd)]
        print("Nb videos :",len(self.videoPathLists))

        self.framesDict = {}
        self.targetDict = {}

        self.removeUselessSeq = removeUselessSeq

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.preproc = transforms.Compose([transforms.Resize(imgSize),transforms.ToTensor(),normalize])
        self.preprocAudio = transforms.ToTensor()

    def __iter__(self):

        seqList = []
        for videoPath in self.videoPathLists:

            vidName = os.path.splitext(os.path.basename(videoPath))[0]

            self.targetDict[vidName] = torch.tensor(np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(self.dataset,vidName)))

            if not vidName in self.framesDict:
                self.framesDict[vidName]= np.array(sorted(glob.glob(os.path.splitext(videoPath)[0]+"/middleFrames/frame*.png"),key=modelBuilder.findNumbers))

            frameInd = 0
            frameNb = len(self.framesDict[vidName])

            seqLengths = np.random.randint(self.lMin,self.lMax+1,size=(frameNb//self.lMin)+(frameNb%self.lMin!=0))

            seqInds = np.cumsum(seqLengths)
            seqInds = np.concatenate(([0],seqInds),axis=0)

            #The number of the last sequence made with the video
            lastSeqNb = np.where((seqInds >= frameNb-self.lMin))[0][0]

            seqInds[-1] = frameNb

            if self.shuffle:
                seqInds = seqInds[:lastSeqNb+1]

            starts = seqInds[:-1]
            ends = seqInds[1:]

            if self.removeUselessSeq:
                starts,ends = self.removeUselessSeqFunc(starts,ends,self.targetDict[vidName])

            if ends[-1]-starts[-1]+1 > self.lMax:
                ends[-1] = starts[-1] + self.lMax -1

            lengs = np.array(list(map(lambda x: x[1]-x[0]+1,list(zip(starts,ends)))))

            names = np.array(vidName)[np.newaxis].repeat(len(seqInds)-1)

            seqList.extend([{"vidName":vidName,"start":start,"end":end} for vidName,start,end in zip(names,starts,ends)])


        self.seqList = np.array(seqList,dtype=object)

        if self.shuffle:
            np.random.shuffle(self.seqList)
        self.currInd = 0

        self.batchNb = len(self.seqList)//self.batchSize

        return self

    def __next__(self):

        if self.currInd >= len(self.seqList):
            raise StopIteration

        batchSize = min(self.batchSize,len(self.seqList)-self.currInd)

        if self.visualFeat:
            videoTensor = torch.zeros((batchSize,self.lMax,3,self.imgSize[0],self.imgSize[1]))
        else:
            videoTensor = None

        if self.audioFeat:
            audioTensor = torch.zeros((batchSize,self.lMax,1,int(self.audioLen*96),64))
        else:
            audioTensor = None

        if (not self.visualFeat) and (not self.audioFeat):
            raise ValueError("At least one modality should be chosen among visual and audio")

        targetTensor = torch.zeros((batchSize,self.lMax))
        seqLenTensor = np.zeros((batchSize)).astype(int)
        imagePathArray = []
        i=0

        seqList = self.seqList[self.currInd:self.currInd+batchSize]

        for i,seq in enumerate(seqList):

            imagePathArray.append(self.framesDict[seq["vidName"]][seq["start"]:seq["end"]])

            if self.visualFeat:
                vidTens = torch.cat(list(map(lambda x:self.preproc(Image.open(x)).unsqueeze(0),self.framesDict[seq["vidName"]][seq["start"]:seq["end"]])),dim=0)
                videoTensor[i,:len(vidTens)] = vidTens
                sequenceLen = len(vidTens)

            if self.audioFeat:
                audioTens = torch.cat(list(map(lambda x:self.preprocAudio(vggish_input.wavfile_to_examples(x[:-3]+"wav")).permute(1,2,0).unsqueeze(0),self.framesDict[seq["vidName"]][seq["start"]:seq["end"]])),dim=0)
                audioTensor[i,:len(audioTens)] = audioTens
                sequenceLen = len(audioTens)

            targs = self.targetDict[seq["vidName"]][seq["start"]:seq["end"]]
            if targs[0] == 1:
                targs[0] = 0
            targetTensor[i,:sequenceLen] = targs
            seqLenTensor[i] = sequenceLen


        self.currInd += self.batchSize

        return videoTensor,audioTensor,targetTensor,torch.tensor(seqLenTensor),imagePathArray

    def removeUselessSeqFunc(self,starts,ends,targets):

        starts_filt = []
        ends_filt = []

        for start,end in zip(starts,ends):
            if targets[start] == 1:
                targets[start] = 0

            if targets[start:end+1].sum() > 0:
                starts_filt.append(start)
                ends_filt.append(end)

        return starts_filt,ends_filt

class PairLoader():

    def __init__(self,dataset,batchSize,imgSize,propStart,propEnd,shuffle):

        self.batchSize = batchSize
        self.videoPathLists = list(filter(lambda x:x.find(".wav") ==-1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        nbVid = len(self.videoPathLists)
        self.videoPathLists = self.videoPathLists[int(nbVid*propStart):int(nbVid*propEnd)]
        print("Nb videos :",len(self.videoPathLists))
        self.dataset=dataset
        self.targetDict = {}
        self.shuffle = shuffle
        for videoPath in self.videoPathLists:
            vidName = os.path.basename(os.path.splitext(videoPath)[0])
            self.targetDict[vidName] = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preproc = transforms.Compose([transforms.Resize(imgSize),transforms.ToTensor(),normalize])

    def __iter__(self):

        self.vidNames = []
        self.anchList,self.posList,self.negList = [],[],[]
        self.anchTargList,self.posTargList,self.negTargList = [],[],[]

        for videoPath in self.videoPathLists:
            vidName = os.path.basename(os.path.splitext(videoPath)[0])

            imagePathList = list(filter(lambda x:x.find(".wav") ==-1,sorted(glob.glob("../data/{}/{}/middleFrames/*.*".format(self.dataset,vidName)))))
            sceneInds = np.cumsum(self.targetDict[vidName][:len(imagePathList)])

            #Building anchor tensor
            anchList,anchTargList = np.array(imagePathList),sceneInds
            anchorAndTarget = np.concatenate((anchList[:,np.newaxis],anchTargList[:,np.newaxis].astype(str)),axis=1)

            #Building positive tensor
            posAndTarget = np.zeros_like(anchorAndTarget)
            for i in range(int(anchTargList[-1])+1):
                imageTargScene = anchorAndTarget[anchorAndTarget[:,1].astype(float) == i]
                np.random.shuffle(imageTargScene)
                posAndTarget[anchorAndTarget[:,1].astype(float) == i] = imageTargScene

            posList,posTargList = posAndTarget.transpose()

            #Buiding negative tensor
            negAndTarget = np.zeros_like(anchorAndTarget)

            for i in range(int(anchTargList[-1])+1):

                imageTargDiff = anchorAndTarget[anchorAndTarget[:,1].astype(float) != i]
                np.random.shuffle(imageTargDiff)

                negAndTarget[anchorAndTarget[:,1].astype(float) == i] = imageTargDiff[:(anchorAndTarget[:,1].astype(float) == i).sum()]

            negList,negTargList = negAndTarget.transpose()

            self.vidNames.extend([vidName for i in range(len(imagePathList))])
            self.anchList.extend(anchList)
            self.posList.extend(posList)
            self.negList.extend(negList)
            self.anchTargList.extend(anchTargList)
            self.posTargList.extend(posTargList)
            self.negTargList.extend(negTargList)

        if self.shuffle:

            zipped = np.concatenate((np.array(self.vidNames)[:,np.newaxis],\
                                     np.array(self.anchList)[:,np.newaxis],np.array(self.posList)[:,np.newaxis],np.array(self.negList)[:,np.newaxis],\
                                     np.array(self.anchTargList)[:,np.newaxis],np.array(self.posTargList)[:,np.newaxis],np.array(self.negTargList)[:,np.newaxis]),axis=1)

            np.random.shuffle(zipped)
            self.vidNames,self.anchList,self.posList,self.negList,self.anchTargList,self.posTargList,self.negTargList = zipped.transpose()

        self.batchNb = len(self.vidNames)//self.batchSize

        self.currInd = 0
        return self

    def __next__(self):

        if self.currInd >= len(self.anchList):
            raise StopIteration

        batchSize = min(self.batchSize,len(self.anchList)-self.currInd)

        videoNames = self.vidNames[self.currInd:self.currInd+batchSize]
        batchTens1 = torch.cat(list(map(lambda x: self.preproc(Image.open(x)).unsqueeze(0),self.anchList[self.currInd:self.currInd+batchSize])),dim=0)
        batchTens2 = torch.cat(list(map(lambda x: self.preproc(Image.open(x)).unsqueeze(0),self.posList[self.currInd:self.currInd+batchSize])),dim=0)
        batchTens3 = torch.cat(list(map(lambda x: self.preproc(Image.open(x)).unsqueeze(0),self.negList[self.currInd:self.currInd+batchSize])),dim=0)
        targ1 = self.anchTargList[self.currInd:self.currInd+batchSize]
        targ2 = self.posTargList[self.currInd:self.currInd+batchSize]
        targ3 = self.negTargList[self.currInd:self.currInd+batchSize]

        self.currInd += self.batchSize

        return videoNames,batchTens1,batchTens2,batchTens3,targ1,targ2,targ3

def main():

    loader = PairLoader("OVSD",1,(299,299),0,1,True)

    for names,tens1,tens2,tens3,targ1,targ2,targ3 in loader:

        print(tens1.sum(),tens2.sum(),tens3.sum(),targ1,targ2,targ3)
        #sys.exit(0)
    getMiddleFrames("OVSD")

if __name__ == "__main__":
    main()
