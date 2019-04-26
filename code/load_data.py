
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
import processResults
import pims
import time
import torchvision
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
'''
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

class TestSeqLoader():

    def __init__(self,evalL,dataset,propStart,propEnd):

        self.evalL = evalL
        self.dataset = dataset
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]

    def __iter__(self):

        self.videoInd = 0
        self.shotInd = 0
        return self
    def __next__(self):

        if self.videoInd == len(self.videoPaths):
            raise StopIteration

        L = self.evalL

        videoPath = self.videoPaths[self.videoInd]
        vidName = os.path.basename(os.path.splitext(videoPath)[0])

        shotPaths = sorted(glob.glob("../data/{}/{}/image/*/*.pth".format(self.dataset,vidName)),key=modelBuilder.findNumbers)
        shotPaths = shotPaths[self.shotInd:self.shotInd+L]

        gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(self.dataset,vidName))
        gt = gt[self.shotInd:self.shotInd+L]

        shots = torch.cat(list(map(lambda x: self.middleFrame(torch.load(x)).unsqueeze(0),shotPaths)),dim=0)

        if self.shotInd + L >= len(shotPaths):
            self.shotInd = 0
            self.videoInd += 1

        return shots,torch.tensor(gt)

    def middleFrame(self,shot):

        return shot[len(shot)//2]

class TrainSeqLoader():

    def __init__(self,batchSize,dataset,propStart,propEnd,lMin,lMax,imgSize):

        self.batchSize = batchSize
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.imgSize = imgSize
        self.nbShots = 0
        self.lMin,self.lMax = lMin,lMax
        self.dataset = dataset
        for videoPath in self.videoPaths:
            videoFold = os.path.splitext(videoPath)[0]
            self.nbShots += len(glob.glob(videoFold+"/image/*/*.pth"))

        print("Number of total shots : ",self.nbShots)

    def __iter__(self):

        self.sumL = 0
        return self
    def __next__(self):

        if self.sumL*self.batchSize > self.nbShots:
            raise StopIteration

        l = np.random.randint(self.lMin,self.lMax+1)
        self.sumL += l

        vidInds = self.nonRepRandInt(len(self.videoPaths),self.batchSize)

        data = torch.zeros(self.batchSize,l,3,self.imgSize[0],self.imgSize[1])
        targ = torch.zeros(self.batchSize,l)

        for i,vidInd in enumerate(vidInds):

            vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

            print("../data/{}/{}/image/*/*.pth".format(self.dataset,vidName))
            shotPaths = np.array(sorted(glob.glob("../data/{}/{}/image/*/*.pth".format(self.dataset,vidName)),key=modelBuilder.findNumbers))
            gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(self.dataset,vidName))

            zipped = np.concatenate((shotPaths[:,np.newaxis],gt[:,np.newaxis]),axis=1)
            np.random.shuffle(zipped)

            shotPaths,gt = zipped[:l].transpose()

            frameSeq = torch.cat(list(map(lambda x:self.sampleAFrame(torch.load(x)).unsqueeze(0),shotPaths)),dim=0)

            if frameSeq.size(1) != 3:
                frameSeq = frameSeq.expand(frameSeq.size(0),3,frameSeq.size(2),frameSeq.size(3))

            data[i] = frameSeq
            targ[i] = torch.tensor(gt.astype(float).astype(int))

        return data,targ

    def sampleAFrame(self,x):
        return x[np.random.randint(0,len(x))]

    def nonRepRandInt(self,nbMax,size):
        ints = np.arange(0,nbMax)
        np.random.shuffle(ints)
        ints = ints[:size]
        return ints
'''
class PairLoader():

    def __init__(self,dataset,batchSize,imgSize,propStart,propEnd,shuffle,audioLen):

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
            self.targetDict[vidName] = getGT(dataset,vidName)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.Resize(imgSize),transforms.ToTensor(),normalize])

        self.audioLen = audioLen


    def __iter__(self):

        self.audioDict = {}
        self.videoDict = {}
        self.fpsDict = {}
        self.fsDict = {}

        self.vidPaths = []
        self.vidNames = []
        self.anchList,self.posList,self.negList = [],[],[]
        self.anchTargList,self.posTargList,self.negTargList = [],[],[]

        for videoPath in self.videoPathLists:
            vidName = os.path.basename(os.path.splitext(videoPath)[0])

            shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
            shotInds = np.arange(len(shotBounds))
            shotBounds = torch.tensor(shotBounds[shotInds.astype(int)]).float()
            frameInds = torch.distributions.uniform.Uniform(shotBounds[:,0], shotBounds[:,1]+1).sample().long()

            #imagePathList = list(filter(lambda x:x.find(".wav") ==-1,sorted(glob.glob("../data/{}/{}/middleFrames/*.*".format(self.dataset,vidName)))))
            sceneInds = np.cumsum(self.targetDict[vidName])

            #Building anchor tensor
            anchList,anchTargList = np.array(frameInds),sceneInds
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

                nbRep = 1+len(negAndTarget[anchorAndTarget[:,1].astype(float) == i])//len(imageTargDiff)

                negAndTarget[anchorAndTarget[:,1].astype(float) == i] = imageTargDiff.repeat(nbRep,axis=0)[:(anchorAndTarget[:,1].astype(float) == i).sum()]

            negList,negTargList = negAndTarget.transpose()

            self.vidPaths.extend([videoPath for i in range(len(frameInds))])
            self.vidNames.extend([vidName for i in range(len(frameInds))])
            self.anchList.extend(anchList)
            self.posList.extend(posList)
            self.negList.extend(negList)
            self.anchTargList.extend(anchTargList)
            self.posTargList.extend(posTargList)
            self.negTargList.extend(negTargList)

        self.zipped = np.concatenate((np.array(self.vidPaths)[:,np.newaxis],np.array(self.vidNames)[:,np.newaxis],\
                                 np.array(self.anchList)[:,np.newaxis],np.array(self.posList)[:,np.newaxis],np.array(self.negList)[:,np.newaxis],\
                                 np.array(self.anchTargList)[:,np.newaxis],np.array(self.posTargList)[:,np.newaxis],np.array(self.negTargList)[:,np.newaxis]),axis=1)

        if self.shuffle:

            np.random.shuffle(self.zipped)

        _,self.vidNames,self.anchList,self.posList,self.negList,self.anchTargList,self.posTargList,self.negTargList = self.zipped.transpose()

        self.batchNb = len(self.vidNames)//self.batchSize

        self.currInd = 0
        return self

    def __next__(self):

        if self.currInd >= len(self.anchList):
            raise StopIteration

        batchSize = min(self.batchSize,len(self.anchList)-self.currInd)

        videoNames = self.vidNames[self.currInd:self.currInd+batchSize]

        audioTens1 = torch.cat(list(map(lambda x: self.readAudio(x,2),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
        audioTens2 = torch.cat(list(map(lambda x: self.readAudio(x,3),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
        audioTens3 = torch.cat(list(map(lambda x: self.readAudio(x,4),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)

        batchTens1 = torch.cat(list(map(lambda x: self.readImage(x,2),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
        batchTens2 = torch.cat(list(map(lambda x: self.readImage(x,3),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)
        batchTens3 = torch.cat(list(map(lambda x: self.readImage(x,4),self.zipped[self.currInd:self.currInd+batchSize])),dim=0)

        targ1 = self.anchTargList[self.currInd:self.currInd+batchSize]
        targ2 = self.posTargList[self.currInd:self.currInd+batchSize]
        targ3 = self.negTargList[self.currInd:self.currInd+batchSize]

        self.currInd += self.batchSize

        return videoNames,batchTens1,batchTens2,batchTens3,\
                          audioTens1,audioTens2,audioTens3,\
                          torch.tensor(targ1.astype(int)),torch.tensor(targ2.astype(int)),torch.tensor(targ3.astype(int))

    def readImage(self,x,i):

        if not x[0] in self.videoDict.keys():
            self.videoDict[x[0]] = pims.Video(x[0])

        #print(x)
        #print(self.videoDict[x[0]])
        #print(x[0])

        return self.preproc(self.videoDict[x[0]][int(x[i])]).unsqueeze(0)

    def readAudio(self,x,i):

        audioPath = os.path.splitext(x[0])[0]+".wav"

        if not audioPath in self.audioDict.keys():
            tree = ET.parse("../data/{}/{}/result.xml".format(self.dataset,x[1])).getroot()
            fps = float(tree.find("content").find("head").find("media").find("fps").text)
            audioData, fs = sf.read(audioPath)
            self.audioDict[audioPath] = audioData
            self.fpsDict[audioPath] = fps
            self.fsDict[audioPath] = fs

        audioData = readAudio(self.audioDict[audioPath],int(x[i]),self.fpsDict[audioPath],self.fsDict[audioPath],self.audioLen)
        return torch.tensor(vggish_input.waveform_to_examples(audioData,self.fsDict[audioPath])/32768.0).unsqueeze(0).float()

class TrainLoader():

    def __init__(self,batchSize,dataset,propStart,propEnd,lMin,lMax,imgSize,audioLen):

        self.batchSize = batchSize
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.imgSize = imgSize
        self.lMin,self.lMax = lMin,lMax
        self.dataset = dataset
        self.audioLen = audioLen

        self.nbShots = 0
        for videoPath in self.videoPaths:
            videoFold = os.path.splitext(videoPath)[0]
            print(videoFold,"../data/{}/{}/result.xml".format(self.dataset,os.path.basename(videoFold)))
            self.nbShots += len(processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,os.path.basename(videoFold))))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.Resize(imgSize),transforms.ToTensor(),normalize])


    def __iter__(self):

        self.sumL = 0
        return self

    def __next__(self):

        if self.sumL*self.batchSize > self.nbShots:
            raise StopIteration

        l = np.random.randint(self.lMin,self.lMax+1)
        self.sumL += l

        vidInds = self.nonRepRandInt(len(self.videoPaths),self.batchSize)

        data = torch.zeros(self.batchSize,l,3,self.imgSize[0],self.imgSize[1])
        audio = torch.zeros(self.batchSize,l,1,int(96*self.audioLen),64)
        targ = torch.zeros(self.batchSize,l)
        vidNames = []

        for i,vidInd in enumerate(vidInds):

            vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

            tree = ET.parse("../data/{}/{}/result.xml".format(self.dataset,vidName)).getroot()
            fps = float(tree.find("content").find("head").find("media").find("fps").text)

            shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
            shotInds = np.arange(len(shotBounds))

            gt = getGT(self.dataset,vidName)

            #The scene number of each shot
            gt = np.cumsum(gt)-1

            #Permutate the scenes
            gt = self.permuteScenes(gt)

            #Select some shots
            zipped = np.concatenate((shotInds[:,np.newaxis],gt[:,np.newaxis]),axis=1)
            np.random.shuffle(zipped)
            zipped = zipped[:l]

            #If the shots are not sorted, each shot is very likely to be followed by a shot from a different scene
            #Sorting them balance this effect.
            zipped = zipped[zipped[:,1].argsort()]

            shotInds,gt = zipped.transpose()

            #A boolean array indicating if a selected shot in preceded by a shot from a different scene
            gt[1:] = (gt[1:] !=  gt[:-1])
            gt[0] = 0

            shotBounds = torch.tensor(shotBounds[shotInds.astype(int)]).float()
            frameInds = torch.distributions.uniform.Uniform(shotBounds[:,0], shotBounds[:,1]+1).sample().long()

            video = pims.Video(self.videoPaths[vidInd])
            audioData, fs = sf.read(os.path.splitext(self.videoPaths[vidInd])[0]+".wav")

            arrToExamp = torchvision.transforms.Lambda(lambda x:torch.tensor(vggish_input.waveform_to_examples(x,fs)/32768.0))
            self.preprocAudio = transforms.Compose([arrToExamp])

            frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)
            audioSeq = torch.cat(list(map(lambda x:self.preprocAudio(readAudio(audioData,x,fps,fs,self.audioLen)).unsqueeze(0),np.array(frameInds))))

            data[i] = frameSeq
            audio[i] = audioSeq
            targ[i] = torch.tensor(gt.astype(float).astype(int))
            vidNames.append(vidName)

        return data,audio.float(),targ,vidNames

    def sampleAFrame(self,x):
        return x[np.random.randint(0,len(x))]

    def nonRepRandInt(self,nbMax,size):
        ints = np.arange(0,nbMax)
        np.random.shuffle(ints)
        ints = ints[:size]
        return ints

    def permuteScenes(self,gt):
        randScenePerm = np.random.permutation(int(gt.max()+1))
        gt_perm = np.zeros_like(gt)
        for j in range(len(gt)):
            gt_perm[j] = randScenePerm[gt[j]]
        gt = gt_perm
        return gt

class TestLoader():

    def __init__(self,evalL,dataset,propStart,propEnd,imgSize,audioLen):
        self.evalL = evalL
        self.dataset = dataset
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.Resize(imgSize),transforms.ToTensor(),normalize])
        self.audioLen = audioLen
        self.nbShots =0
        for videoPath in self.videoPaths:
            videoFold = os.path.splitext(videoPath)[0]
            self.nbShots += len(processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,os.path.basename(videoFold))))

    def __iter__(self):
        self.videoInd = 0
        self.shotInd = 0
        self.sumL = 0
        return self

    def __next__(self):

        if self.videoInd == len(self.videoPaths):
            raise StopIteration

        L = self.evalL
        self.sumL += L

        videoPath = self.videoPaths[self.videoInd]
        video = pims.Video(videoPath)
        audioData, fs = sf.read(os.path.splitext(videoPath)[0]+".wav")

        vidName = os.path.basename(os.path.splitext(videoPath)[0])

        tree = ET.parse("../data/{}/{}/result.xml".format(self.dataset,vidName)).getroot()
        fps = float(tree.find("content").find("head").find("media").find("fps").text)

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
        shotInds =  np.arange(self.shotInd,self.shotInd+L)

        frameInds = self.middleFrames(shotBounds[shotInds])

        for j in range(len(frameInds)):
            img = video[frameInds[j]]
            cv2.imwrite('../vis/testImgPims_{}_{}.png'.format(j,shotInds[j]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        arrToExamp = torchvision.transforms.Lambda(lambda x:torch.tensor(vggish_input.waveform_to_examples(x,fs)/32768.0))
        self.preprocAudio = transforms.Compose([arrToExamp])

        frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)
        audioSeq = torch.cat(list(map(lambda x:self.preprocAudio(readAudio(audioData,x,fps,fs,self.audioLen)).unsqueeze(0),np.array(frameInds))))

        gt = getGT(self.dataset,vidName)[self.shotInd:self.shotInd+L]

        if self.shotInd + L >= len(shotBounds):
            self.shotInd = 0
            self.videoInd += 1

        self.shotInd += L

        return frameSeq.unsqueeze(0),audioSeq.unsqueeze(0).float(),torch.tensor(gt).unsqueeze(0),[vidName]

    def middleFrames(self,shotBounds):
        starts = np.concatenate((shotBounds[:,0],[shotBounds[-1,1]]),axis=0)
        return (starts[1:]+starts[:-1])//2

def readAudio(audioData,i,fps,fs,audio_len):
    time = i/fps
    pos = time*fs
    interv = audio_len*fs/2
    sampleToWrite = audioData[int(round(pos-interv)):int(round(pos+interv)),:]
    fullArray = np.zeros((int(round(pos+interv))-int(round(pos-interv)),sampleToWrite.shape[1]))
    fullArray[:len(sampleToWrite)] = sampleToWrite

    return fullArray

def getGT(dataset,vidName):
    if not os.path.exists("../data/{}/annotations/{}_targ.csv".format(dataset,vidName)):

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(dataset,vidName))

        scenesBounds = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,vidName))
        gt = framesToShot(scenesBounds,shotBounds)
        np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName),gt)
    else:
        gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName)).astype(int)

    return gt.astype(int)

def framesToShot(scenesBounds,shotBounds):

    gt = np.zeros(len(shotBounds))
    sceneInd = 0

    gt[np.power(scenesBounds[:,0][np.newaxis,:]-shotBounds[:,0][:,np.newaxis],2).argmin(axis=0)] = 1
    gt[0] = 0

    return gt

def main():

    #getMiddleFrames("OVSD")

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


    '''
    trainLoad = TrainSeqLoader(1,"OVSD",0,0.5,10,20,(299,299))
    for i,(data,targ) in enumerate(trainLoad):

        print(type(data))
        print(data.size(),targ.size())

        for j,img in enumerate(data[0]):

            img = 255*np.array(((img-img.min())/(img.max()-img.min())).permute(1,2,0))

            cv2.imwrite("../vis/trainVis_{}_{}.png".format(i,j),img)

        sys.exit(0)
    '''
    '''
    trainLoad = TrainLoader(1,"OVSD",0,0.5,10,20,(299,299),1)
    for i,(data,audio,targ,vidName) in enumerate(trainLoad):
        print(data.size(),audio.size(),targ.size(),vidName)

        break


    testLoad = TestLoader(20,"OVSD",0,0.5,(299,299),1)
    for i,(data,audio,targ,vidName) in enumerate(testLoad):
        print(data.size(),audio.size(),targ.size(),vidName)

        break

    '''

    pairLoad = PairLoader("OVSD",16,(299,299),0,0.5,True,1)
    for videoNames,anch,pos,neg,anchAudio,posAudio,negAudio,targ1,targ2,targ3 in pairLoad:
        print(videoNames,anch.size(),anchAudio.size(),targ1.size())
        sys.exit(0)

if __name__ == "__main__":
    main()
