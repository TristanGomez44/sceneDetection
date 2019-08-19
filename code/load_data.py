import sys
import glob
import os

import numpy as np
import torch
from torchvision import transforms
import torchvision
import vggish_input

import soundfile as sf
from PIL import Image

import processResults
import pims
import time

import args
import warnings
warnings.filterwarnings('ignore',module=".*av.*")

import utils

class Sampler(torch.utils.data.sampler.Sampler):
    """ The sampler for the SeqTrDataset dataset
    """

    def __init__(self, nb_videos,nbTotalShots,nbShotPerSeq):
        self.nb_videos = nb_videos

        self.length = nbTotalShots//nbShotPerSeq

    def __iter__(self):

        if self.length > 0:
            return iter(torch.randint(self.nb_videos,(self.length,)))
        else:
            return iter([])
    def __len__(self):
        return self.length

class AdvSampler(torch.utils.data.sampler.Sampler):
    """ The sampler for the SeqTrDataset dataset
    """

    def __init__(self,nbVideos,nbTotalShots,batchSize,nbShotPerSeq):

        self.length = batchSize*nbTotalShots//nbShotPerSeq
        self.nbVideos = nbVideos

    def __iter__(self):

        return iter(torch.randint(self.nbVideos,(self.length,)))

    def __len__(self):
        return self.length

def collateSeq(batch):

    res = list(zip(*batch))

    res[0] = torch.cat(res[0],dim=0)
    if not res[1][0] is None:
        res[1] = torch.cat(res[1],dim=0)

    if torch.is_tensor(res[2][0]):
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
    - imgSize (int): the size of each side of the image
    - audioLen (int): the length of the audio extract for each shot
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - exp_id (str): the name of the experience
    - max_shots (int): the total number of shot to process before stopping the training epoch. The youtube_large dataset contains a million shots, which is why this argument is useful.
    '''

    def __init__(self,dataset,propStart,propEnd,lMin,lMax,imgSize,audioLen,resizeImage,exp_id,max_shots,avgSceneLen):

        super(SeqTrDataset, self).__init__()

        self.videoPaths = findVideos(dataset,propStart,propEnd)

        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.imgSize = imgSize
        self.lMin,self.lMax = lMin,lMax
        self.dataset = dataset
        self.audioLen = audioLen
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
                self.nbShots += len(np.genfromtxt("../data/{}/{}/result.csv".format(self.dataset,os.path.basename(videoFold))))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.RandomResizedCrop(imgSize),transforms.ToTensor(),normalize])

        self.FPSDict = {}

    def __len__(self):
        return self.nbShots

    def __getitem__(self,vidInd):

        data = torch.zeros(self.lMax,3,self.imgSize,self.imgSize)
        targ = torch.zeros(self.lMax)
        vidNames = []

        vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

        if not self.videoPaths[vidInd] in self.FPSDict.keys():
            self.FPSDict[self.videoPaths[vidInd]] = utils.getVideoFPS(self.videoPaths[vidInd])

        fps = utils.getVideoFPS(self.videoPaths[vidInd])

        shotBounds = np.genfromtxt("../data/{}/{}/result.csv".format(self.dataset,vidName))
        shotInds = np.arange(len(shotBounds))

        #Computes the scene number of each shot
        gt = getGT(self.dataset,vidName)
        gt = np.cumsum(gt)

        #Permutate the scene indexes. Later the shots will be sorted and this ensures the shots
        #from scene n do not always appear before shots from scene m > n.
        gt = self.permuteScenes(gt)

        if self.avgSceneLen != -1:
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

        if self.avgSceneLen != -1:
            #Removing shots from scenes that are not chosen
            zippedFiltered = list(filter(lambda x: x[1] in sceneInds,zipped))

            #Prevent a bug where there happen to be zero shots
            if len(zippedFiltered) > 0:
                zipped = zippedFiltered

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
        frameInds = torch.distributions.uniform.Uniform(shotBounds[:,0], shotBounds[:,1]+1).sample((1,))

        #Clamp the maximal value to the last frame index
        lastFrameInd = np.genfromtxt("../data/{}/{}/result.csv".format(self.dataset,vidName)).max()
        frameInds = torch.clamp(frameInds,0,lastFrameInd).long()
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
            print(vidName,frameInds.max(),np.genfromtxt("../data/{}/{}/result.csv".format(self.dataset,vidName)).max(),gt.max())
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

class VideoFrameDataset(torch.utils.data.Dataset):

    def __init__(self,dataset,propStart,propEnd,imgSize,resizeImage,max_shots):

        super(VideoFrameDataset, self).__init__()

        self.videoPaths = findVideos(dataset,propStart,propEnd)
        #np.random.shuffle(self.videoPaths)

        self.imgSize = imgSize
        self.dataset = dataset
        self.nbShots = 0

        #Computing shots
        if max_shots != -1:
            self.nbShots = max_shots
        else:
            for videoPath in self.videoPaths:
                videoFold = os.path.splitext(videoPath)[0]
                self.nbShots += len(np.genfromtxt("{}/result.csv".format(videoFold)))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not resizeImage:
            self.preproc = transforms.Compose([transforms.ToTensor(),normalize])
        else:
            self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.RandomResizedCrop(imgSize),transforms.ToTensor(),normalize])

        self.FPSDict = {}
    def __len__(self):
        return self.nbShots
    def __getitem__(self,vidInd):

        data = torch.zeros(3,self.imgSize,self.imgSize)
        vidFold = os.path.splitext(self.videoPaths[vidInd])[0]
        vidName = os.path.basename(vidFold)

        if not self.videoPaths[vidInd] in self.FPSDict.keys():
            self.FPSDict[self.videoPaths[vidInd]] = utils.getVideoFPS(self.videoPaths[vidInd])

        fps = self.FPSDict[self.videoPaths[vidInd]]

        shotBounds = np.genfromtxt("{}/result.csv".format(vidFold))
        shotInd = np.random.randint(len(shotBounds))

        #Selecting frame indexes
        shotBounds = torch.tensor(shotBounds[shotInd]).float()
        frameInd = torch.distributions.uniform.Uniform(shotBounds[0], shotBounds[1]+1).sample().long()

        video = pims.Video(self.videoPaths[vidInd])

        gt = [1]

        return self.preproc(video[frameInd.item()]).unsqueeze(0),torch.tensor(gt).float().unsqueeze(0),[vidName]

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
    - exp_id (str): the name of the experience
    '''

    def __init__(self,evalL,dataset,propStart,propEnd,imgSize,audioLen,resizeImage,exp_id,randomFrame):
        self.evalL = evalL
        self.dataset = dataset

        self.videoPaths = findVideos(dataset,propStart,propEnd)

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
            self.nbShots += len(np.genfromtxt("../data/{}/{}/result.csv".format(self.dataset,os.path.basename(videoFold))))

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

        fps = utils.getVideoFPS(videoPath)

        shotBounds = np.genfromtxt("../data/{}/{}/result.csv".format(self.dataset,vidName))
        shotInds =  np.arange(self.shotInd,min(self.shotInd+L,len(shotBounds)))

        if not self.randomFrame:
            frameInds = self.regularlySpacedFrames(shotBounds[shotInds]).reshape(-1)
        else:
            shotBoundsToUse = torch.tensor(shotBounds[shotInds.astype(int)]).float()
            frameInds = torch.distributions.uniform.Uniform(shotBoundsToUse[:,0], shotBoundsToUse[:,1]+1).sample((1,)).long()
            frameInds = frameInds.transpose(dim0=0,dim1=1)
            frameInds = np.array(frameInds.contiguous().view(-1))

        arrToExamp = torchvision.transforms.Lambda(lambda x:torch.tensor(vggish_input.waveform_to_examples(x,fs)/32768.0))
        self.preprocAudio = transforms.Compose([arrToExamp])

        try:
            frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)
        except IndexError:
            print(vidName,"max frame",frameInds.max(),np.genfromtxt("../data/{}/{}/result.csv".format(self.dataset,vidName)).max())
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

        frameInds = ((np.arange(1)/1)[np.newaxis,:]*(shotBounds[:,1]-shotBounds[:,0])[:,np.newaxis]+shotBounds[:,0][:,np.newaxis]).astype(int)

        return frameInds

def buildSeqTrainLoader(args,audioLen):

    train_dataset = SeqTrDataset(args.dataset_train,args.train_part_beg,args.train_part_end,args.l_min,args.l_max,\
                                        args.img_size,audioLen,args.resize_image,args.exp_id,args.max_shots,args.avg_scene_len)
    sampler = Sampler(len(train_dataset.videoPaths),train_dataset.nbShots,args.l_max)
    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,sampler=sampler, collate_fn=collateSeq, # use custom collate function here
                      pin_memory=False,num_workers=args.num_workers)

    return trainLoader,train_dataset

def buildFrameTrainLoader(args):

    train_dataset = VideoFrameDataset(args.dataset_train,args.train_part_beg,args.train_part_end,args.img_size,args.resize_image,args.max_shots)
    adv_dataset = VideoFrameDataset(args.dataset_adv,args.adv_part_beg,args.adv_part_end,args.img_size,args.resize_image,args.max_shots)

    nbVideosTr,nbVideosAdv = len(train_dataset.videoPaths),len(adv_dataset.videoPaths)

    #The number of shot is increased by nbVideosTr//nbVideosAdv to make as much iterations as with the Sequence loader
    sampler = AdvSampler(nbVideosAdv,train_dataset.nbShots,args.adv_batch_size,args.l_max)
    trainLoader = torch.utils.data.DataLoader(dataset=adv_dataset,batch_size=args.adv_batch_size,sampler=sampler, collate_fn=collateSeq, # use custom collate function here
                      pin_memory=False,num_workers=args.num_workers)

    return trainLoader

def findVideos(dataset,propStart,propEnd):

    videoPaths = sorted(glob.glob("../data/{}/*.*".format(dataset)))
    videoPaths = list(filter(lambda x:x.find(".wav") == -1,videoPaths))
    videoPaths = list(filter(lambda x:os.path.isfile(x),videoPaths))
    videoPaths = np.array(videoPaths)[int(propStart*len(videoPaths)):int(propEnd*len(videoPaths))]

    return videoPaths

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

        shotBounds = np.genfromtxt("../data/{}/{}/result.csv".format(dataset,vidName))

        scenesBounds = np.genfromtxt("../data/{}/annotations/{}_scenes.txt".format(dataset,vidName))
        gt = utils.framesToShot(scenesBounds,shotBounds)
        np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName),gt)
    else:
        gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName))

    return gt.astype(int)

def addArgs(argreader):

    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                        help='The network producing the features can only be pretrained on \'imageNet\'. This argument must be \
                            set to \'imageNet\' datasets.')
    argreader.parser.add_argument('--batch_size', type=int,metavar='BS',
                        help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int,metavar='BS',
                        help='The batchsize to use for validation')
    argreader.parser.add_argument('--adv_batch_size', type=int,metavar='BS',
                        help='The batchsize to use for adversarial loss')

    argreader.parser.add_argument('--l_min', type=int,metavar='LMIN',
                        help='The minimum length of a training sequence')
    argreader.parser.add_argument('--l_max', type=int,metavar='LMAX',
                        help='The maximum length of a training sequence')
    argreader.parser.add_argument('--val_l', type=int,metavar='LMAX',
                        help='Length of sequences for validation.')
    argreader.parser.add_argument('--dataset_train', type=str, metavar='N',help='the dataset to train. Can be \'OVSD\', \'bbc\',\'bbc2\', \'youtube_large\', \'Holly2\'.')
    argreader.parser.add_argument('--dataset_val', type=str, metavar='N',help='the dataset to validate. Can have the same values as --dataset_train.')
    argreader.parser.add_argument('--dataset_test', type=str, metavar='N',help='the dataset to testing. Can have the same values as --dataset_train.')
    argreader.parser.add_argument('--dataset_adv', type=str, metavar='N',help='the dataset for adversarial loss. A MLP will try to discriminate images \
                                                                                coming from the training dataset (--dataset_train) and from this dataset.\
                                                                                This is ignored if no adversarial loss is to be used (i.e. --adv_weight is set to 0 (see \
                                                                                trainVal.py)). It can have the same values as --dataset_train.')

    argreader.parser.add_argument('--img_size', type=int,metavar='WIDTH',
                        help='The size of each edge of the resized images, if resize_image is True, else, the size of the image')

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
    argreader.parser.add_argument('--adv_part_beg', type=float,metavar='START',
                        help='The (normalized) start position of the dataset to use for adversarial loss')
    argreader.parser.add_argument('--adv_part_end', type=float,metavar='END',
                        help='The (normalized) end position of the dataset to use for adversarial loss')

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
