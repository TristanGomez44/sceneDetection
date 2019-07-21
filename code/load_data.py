
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
import args
import warnings
warnings.filterwarnings('ignore',module=".*av.*")

class Sampler(torch.utils.data.sampler.Sampler):
    """ The sampler for the SeqTrDataset dataset
    """

    def __init__(self, nb_videos,nbTotalShots,nbShotPerSeq,hmIndList=None,hmProp=0):
        self.nb_videos = nb_videos

        self.length = nbTotalShots//nbShotPerSeq

        self.hmIndList = hmIndList
        self.hmProp = hmProp

    def __iter__(self):

        if self.length > 0:
            if (not self.hmIndList is None) and self.hmProp>0:

                self.hmIndList = torch.tensor(self.hmIndList).long()
                self.hmIndList = self.hmIndList[torch.randint(int(self.hmProp*len(self.hmIndList)),(int(self.hmProp*self.length),))]

                self.randIndList = torch.randint(self.nb_videos,(int((1-self.hmProp)*self.length),))

                self.indList = torch.cat((self.hmIndList,self.randIndList),dim=0)

                self.indList = self.indList.numpy()

                np.random.shuffle(self.indList)

                return iter(self.indList)
            else:
                return iter(torch.randint(self.nb_videos,(self.length,)))
        else:
            return iter([])
    def __len__(self):
        return self.length

def collateSeq(batch):

    res = list(zip(*batch))

    res[0] = torch.cat(res[0],dim=0)
    if not res[1][0] is None:
        res[1] = torch.cat(res[1],dim=0)
    res[2] = torch.cat(res[2],dim=0)

    return res

class SeqTrDataset(torch.utils.data.Dataset):
    '''
    The dataset to sample sequence of frames from videos

    When the method __getitem__(i) is called, the dataset select some shots from the video i, select frame(s) from each shot, shuffle them, regroup them by scene and
    returns them. It can also returns the MFCC coefficient of a sound extract for each shot.

    Args:
    - dataset (str): the name of the dataset to use
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - lMin (int): the minimum length of a sequence during training
    - lMax (int): the maximum length of a sequence during training
    - imgSize (tuple): a tuple containing (in order) the width and size of the image
    - audioLen (int): the length of the audio extract for each shot
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - framesPerShot (int): the number of frames to use to reprensent each shot
    - exp_id (str): the name of the experience
    - max_shots (int): the total number of shot to process before stopping the training epoch. The youtube_large dataset contains a million shots, which is why this argument is useful.
    '''

    def __init__(self,dataset,propStart,propEnd,lMin,lMax,imgSize,audioLen,resizeImage,framesPerShot,exp_id,max_shots,avgSceneLen):

        super(SeqTrDataset, self).__init__()

        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = list(filter(lambda x:x.find(".xml") == -1,self.videoPaths))
        self.videoPaths = list(filter(lambda x:os.path.isfile(x),self.videoPaths))

        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.imgSize = imgSize
        self.lMin,self.lMax = lMin,lMax
        self.dataset = dataset
        self.audioLen = audioLen
        self.framesPerShot = framesPerShot
        self.nbShots = 0
        self.exp_id = exp_id
        self.avgSceneLen = avgSceneLen

        if propStart == propEnd:
            self.nbShots = 0
        elif max_shots != -1:
            self.nbShots = max_shots
        else:
            for videoPath in self.videoPaths:
                videoFold = os.path.splitext(videoPath)[0]
                self.nbShots += len(processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,os.path.basename(videoFold))))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.RandomResizedCrop(imgSize),transforms.ToTensor(),normalize])

        self.FPSDict = {}

    def __len__(self):
        return self.nbShots

    def __getitem__(self,vidInd):

        data = torch.zeros(self.framesPerShot*self.lMax,3,self.imgSize[0],self.imgSize[1])
        targ = torch.zeros(self.lMax)
        vidNames = []

        vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

        if not self.videoPaths[vidInd] in self.FPSDict.keys():
            self.FPSDict[self.videoPaths[vidInd]] = processResults.getVideoFPS(self.videoPaths[vidInd],self.exp_id)

        fps = processResults.getVideoFPS(self.videoPaths[vidInd],self.exp_id)

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
        shotInds = np.arange(len(shotBounds))

        #Computes the scene number of each shot
        gt = getGT(self.dataset,vidName)
        gt = np.cumsum(gt)

        #Permutate the scene indexes. Later the shots will be sorted and this ensures the shots
        #from scene n do not always appear before shots from scene m > n.
        gt = self.permuteScenes(gt)

        #Choosing some scenes
        nbScenes = len(np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(self.dataset,vidName)))
        sceneInds = np.arange(nbScenes)
        np.random.shuffle(sceneInds)
        nbChosenScenes = self.lMax//self.avgSceneLen

        sceneInds = sceneInds[:nbChosenScenes]
        #This try/except block captures the error that raises when the shotInds and the gt
        #do not have the same length. This can happen if the data is badly formated
        #which would indicate a missed case in the formatData.py script.
        try:
            zipped = np.concatenate((shotInds[:,np.newaxis],gt[:,np.newaxis]),axis=1)
        except ValueError:
            print("Error : ",vidName,"len(shotInds)",len(shotInds),"len(gt)",len(gt))
            sys.exit(0)

        #Removing shots from scenes that are not chosen
        zipped = list(filter(lambda x: x[1] in sceneInds,zipped))
        #Select some shots
        np.random.shuffle(zipped)
        zipped = np.array(zipped)[:self.lMax]

        #Some shot will be repeated if there not enough to make a lMax long sequence
        if len(zipped) < self.lMax:
            repeatedShotInd = np.random.randint(len(zipped),size=self.lMax-len(zipped))
            zipped = np.concatenate((zipped,zipped[repeatedShotInd]),axis=0)

        #If the shots are not sorted, each shot is very likely to be followed by a shot from a different scene
        #Sorting them balance this effect.
        zipped = zipped[zipped[:,1].argsort()]

        shotInds,gt = zipped.transpose()

        #A boolean array indicating if a selected shot in preceded by a shot from a different scene
        gt[1:] = (gt[1:] !=  gt[:-1])
        gt[0] = 0

        #Selecting frame indexes
        shotBounds = torch.tensor(shotBounds[shotInds.astype(int)]).float()
        frameInds = torch.distributions.uniform.Uniform(shotBounds[:,0], shotBounds[:,1]+1).sample((self.framesPerShot,)).long()

        frameInds = frameInds.transpose(dim0=0,dim1=1)
        frameInds = frameInds.contiguous().view(-1)

        video = pims.Video(self.videoPaths[vidInd])

        arrToExamp = torchvision.transforms.Lambda(lambda x:torch.tensor(vggish_input.waveform_to_examples(x,fs)/32768.0))
        self.preprocAudio = transforms.Compose([arrToExamp])

        #Building the frame sequence
        #This try/except block captures the error that raises when a frame index is too high.
        #This can happen if the data is badly formated which would indicate a missed case in the formatData.py script.
        try:
            frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)
        except IndexError:
            print(vidName,frameInds.max(),processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName)).max(),gt.max())
            sys.exit(0)

        if self.audioLen > 0:
            #Build the audio sequence
            audioData, fs = sf.read(os.path.splitext(self.videoPaths[vidInd])[0]+".wav")
            audioSeq = torch.cat(list(map(lambda x:self.preprocAudio(readAudio(audioData,x,fps,fs,self.audioLen)).unsqueeze(0),np.array(frameInds))))
            audioSeq = audioSeq.unsqueeze(0).float()
        else:
            audioSeq = None

        return frameSeq.unsqueeze(0),audioSeq,torch.tensor(gt).float().unsqueeze(0),vidName

    def permuteScenes(self,gt):
        randScenePerm = np.random.permutation(int(gt.max()+1))
        gt_perm = np.zeros_like(gt)
        for j in range(len(gt)):
            gt_perm[j] = randScenePerm[gt[j]]
        gt = gt_perm
        return gt

class TestLoader():
    '''
    The dataset to sample sequence of frames from videos. As the video contains a great number of shot,
    each video is processed through several batches and each batch contain only one sequence.

    Args:
    - evalL (int): the length of a sequence in a batch. A big value will reduce the number of batches necessary to process a whole video
    - dataset (str): the name of the dataset
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - imgSize (tuple): a tuple containing (in order) the width and size of the image
    - audioLen (int): the length of the audio extract for each shot
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - framesPerShot (int): the number of frames to use to reprensent each shot
    - exp_id (str): the name of the experience
    '''

    def __init__(self,evalL,dataset,propStart,propEnd,imgSize,audioLen,resizeImage,framesPerShot,exp_id,randomFrame):
        self.evalL = evalL
        self.dataset = dataset
        self.videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/{}/*.*".format(dataset)))))
        self.videoPaths = list(filter(lambda x:x.find(".xml") == -1,self.videoPaths))
        self.videoPaths = list(filter(lambda x:os.path.isfile(x),self.videoPaths))
        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.framesPerShot = framesPerShot
        self.randomFrame = randomFrame

        self.exp_id = exp_id

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.CenterCrop(imgSize),transforms.ToTensor(),normalize])

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

        vidName = os.path.basename(os.path.splitext(videoPath)[0])

        fps = processResults.getVideoFPS(videoPath,self.exp_id)

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName))
        shotInds =  np.arange(self.shotInd,min(self.shotInd+L,len(shotBounds)))

        if not self.randomFrame:
            frameInds = self.regularlySpacedFrames(shotBounds[shotInds]).reshape(-1)
        else:
            shotBoundsToUse = torch.tensor(shotBounds[shotInds.astype(int)]).float()
            frameInds = torch.distributions.uniform.Uniform(shotBoundsToUse[:,0], shotBoundsToUse[:,1]+1).sample((self.framesPerShot,)).long()
            frameInds = frameInds.transpose(dim0=0,dim1=1)
            frameInds = np.array(frameInds.contiguous().view(-1))

        arrToExamp = torchvision.transforms.Lambda(lambda x:torch.tensor(vggish_input.waveform_to_examples(x,fs)/32768.0))
        self.preprocAudio = transforms.Compose([arrToExamp])

        try:
            frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)
        except IndexError:
            print(vidName,"max frame",frameInds.max(),processResults.xmlToArray("../data/{}/{}/result.xml".format(self.dataset,vidName)).max())
            sys.exit(0)

        if self.audioLen > 0:
            audioData, fs = sf.read(os.path.splitext(videoPath)[0]+".wav")
            audioSeq = torch.cat(list(map(lambda x:self.preprocAudio(readAudio(audioData,x,fps,fs,self.audioLen)).unsqueeze(0),np.array(frameInds))))
            audioSeq = audioSeq.unsqueeze(0).float()
        else:
            audioSeq = None

        gt = getGT(self.dataset,vidName)[self.shotInd:min(self.shotInd+L,len(shotBounds))]

        if shotInds[-1] + 1 == len(shotBounds):
            self.shotInd = 0
            self.videoInd += 1
        else:
            self.shotInd += L

        return frameSeq.unsqueeze(0),audioSeq,torch.tensor(gt).float().unsqueeze(0),vidName,torch.tensor(frameInds).int()

    def regularlySpacedFrames(self,shotBounds):
        ''' Select several frame indexs regularly spaced in each shot '''

        frameInds = ((np.arange(self.framesPerShot)/self.framesPerShot)[np.newaxis,:]*(shotBounds[:,1]-shotBounds[:,0])[:,np.newaxis]+shotBounds[:,0][:,np.newaxis]).astype(int)

        return frameInds

def readAudio(audioData,i,fps,fs,audio_len):
    ''' Select part of an audio track
    Args:
    - audioData (array) the full waveform
    - i (int) the frame index at which the extract should be centered
    - fps (int): the number of frams per second
    - audio_len (float): the length of the audio extract, in seconds

    Returns:
    - fullArray (array): the audio extract

    '''

    time = float(i)/float(fps)
    pos = time*fs
    interv = audio_len*fs/2
    sampleToWrite = audioData[int(round(pos-interv)):int(round(pos+interv)),:]
    fullArray = np.zeros((int(round(pos+interv))-int(round(pos-interv)),sampleToWrite.shape[1]))
    fullArray[:len(sampleToWrite)] = sampleToWrite

    return fullArray

def getGT(dataset,vidName):
    ''' For one video, returns a list of 0,1 indicating for each shot if it's the first shot of a scene or not.

    This function computes the 0,1 list and save if it does not already exists.

    Args:
    - dataset (str): the dataset name
    - vidName (str): the video name. It is the name of the videol file minux the extension.
    Returns:
    - gt (array): a list of 0,1 indicating for each shot if it's the first shot of a scene or not.

    '''

    if not os.path.exists("../data/{}/annotations/{}_targ.csv".format(dataset,vidName)):

        shotBounds = processResults.xmlToArray("../data/{}/{}/result.xml".format(dataset,vidName))

        scenesBounds = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,vidName))
        gt = framesToShot(scenesBounds,shotBounds)
        np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName),gt)
    else:
        gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName))

    return gt.astype(int)

def framesToShot(scenesBounds,shotBounds):
    ''' Convert a list of scene bounds expressed with frame index into a list of scene bounds expressed with shot index

    Args:
    - scenesBounds (array): the scene bounds expressed with frame index
    - shotBounds (array): the shot bounds expressed with frame index

    Returns:
    - gt (array): the scene bounds expressed with shot index

    '''

    gt = np.zeros(len(shotBounds))
    sceneInd = 0

    gt[np.power(scenesBounds[:,0][np.newaxis,:]-shotBounds[:,0][:,np.newaxis],2).argmin(axis=0)] = 1
    gt[0] = 0

    return gt

def addArgs(argreader):

    argreader.parser.add_argument('--hm_prop', type=float, metavar='N',
                        help='Proportion of videos that will be re-used during next epoch for hard-mining.')
    argreader.parser.add_argument('--epochs_hm', type=int, metavar='N',
                        help='The number of epochs to wait before updating the hard mined videos')
    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                        help='The network producing the features can be either pretrained on \'imageNet\' or \'places365\'. This argument \
                            selects one of the two datasets.')
    argreader.parser.add_argument('--batch_size', type=int,metavar='BS',
                        help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int,metavar='BS',
                        help='The batchsize to use for validation')
    argreader.parser.add_argument('--l_min', type=int,metavar='LMIN',
                        help='The minimum length of a training sequence')
    argreader.parser.add_argument('--l_max', type=int,metavar='LMAX',
                        help='The maximum length of a training sequence')
    argreader.parser.add_argument('--val_l', type=int,metavar='LMAX',
                        help='Length of sequences for validation.')
    argreader.parser.add_argument('--dataset_train', type=str, metavar='N',help='the dataset to train. Can be \'OVSD\', \'PlanetEarth\' or \'RAIDataset\'.')
    argreader.parser.add_argument('--dataset_val', type=str, metavar='N',help='the dataset to validate. Can be \'OVSD\', \'PlanetEarth\' or \'RAIDataset\'.')
    argreader.parser.add_argument('--dataset_test', type=str, metavar='N',help='the dataset to testing. Can be \'OVSD\', \'PlanetEarth\' or \'RAIDataset\'.')
    argreader.parser.add_argument('--img_width', type=int,metavar='WIDTH',
                        help='The width of the resized images, if resize_image is True, else, the size of the image')
    argreader.parser.add_argument('--img_heigth', type=int,metavar='HEIGTH',
                        help='The height of the resized images, if resize_image is True, else, the size of the image')
    argreader.parser.add_argument('--train_part_beg', type=float,metavar='START',
                        help='The (normalized) start position of the dataset to use for training')
    argreader.parser.add_argument('--train_part_end', type=float,metavar='END',
                        help='The (normalized) end position of the dataset to use for training')
    argreader.parser.add_argument('--val_part_beg', type=float,metavar='START',
                        help='The (normalized) start position of the dataset to use for validation')
    argreader.parser.add_argument('--val_part_end', type=float,metavar='END',
                        help='The (normalized) end position of the dataset to use for validation')
    argreader.parser.add_argument('--test_part_beg', type=float,metavar='START',
                        help='The (normalized) start position of the dataset to use for testing')
    argreader.parser.add_argument('--test_part_end', type=float,metavar='END',
                        help='The (normalized) end position of the dataset to use for testing')
    argreader.parser.add_argument('--resize_image', type=args.str2bool, metavar='S',
                        help='to resize the image to the size indicated by the img_width and img_heigth arguments.')
    argreader.parser.add_argument('--max_shots', type=int,metavar='NOTE',
                        help="The maximum number of shots to use during an epoch before validating")
    argreader.parser.add_argument('--audio_len', type=float,metavar='NOTE',
                        help="The length of the audio for each shot (in seconds)")
    argreader.parser.add_argument('--random_frame_val', type=args.str2bool, metavar='N',
                        help='If true, random frames instead of middle frames will be used as key frame during validation. ')

    argreader.parser.add_argument('--avg_scene_len', type=int, metavar='N',
                        help='The average scene length in a training sequence')

    return argreader

def main():

    #getMiddleFrames("OVSD")

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    def testLoader(trainLoader,nbBatch_max=20,countFirst=True):
        nbBatch = 0
        time_start = time.time()
        timeList = np.zeros(nbBatch_max-(not countFirst))
        for batch in trainLoader:

            if not (nbBatch==0 and not countFirst):
                timeList[nbBatch-(not countFirst)] = time.time()-time_start

            nbBatch += 1
            time_start = time.time()
            if nbBatch == nbBatch_max:
                break
        end_time = time.time()

        print("Time per batch : ",timeList.mean(),timeList.std(),timeList)

    train_dataset = SeqTrDataset(4,"Holly2",0,0.9,25,35,(298,298),1,False,1,"TestOfPytorchLoader")
    sampler = Sampler(len(train_dataset.videoPaths))

    for i in [4,8]:
        trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,
                          batch_size=4,
                          sampler=sampler,
                          collate_fn=collateSeq, # use custom collate function here
                          pin_memory=True,
                          num_workers=i)

        print("num_workers",i)
        testLoader(trainLoader,countFirst=False)

    trainLoader = TrainLoader(4,"Holly2",0,0.9,25,35,(298,298),1,False,1,"TestOfPytorchLoader")
    testLoader(trainLoader)

if __name__ == "__main__":
    main()
