from args import ArgReader
from args import str2bool
import modelBuilder
import glob
import numpy as np
import load_data
import torch
import load_data
import os
import sys

import processResults
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from nn import DataParallel

import gc

def baselineAllVids(dataset,batchSize,visualFeat,audioFeat,pretrainSet,cuda,intervToProcess,audioLen):

    #Extract the middle frame first
    load_data.getMiddleFrames(dataset,audioLen)

    vidPaths = np.array(sorted(glob.glob("../data/{}/*/middleFrames/".format(dataset))),dtype=str)[intervToProcess]

    for vidPath in vidPaths:
        print(vidPath)

        imagePathList = np.array(sorted(glob.glob(vidPath+"/*"),key=modelBuilder.findNumbers),dtype=str)
        imagePathList = list(filter(lambda x:x.find(".wav") == -1,imagePathList))

        if audioFeat != "None":
            audioPathList = np.array(sorted(glob.glob(vidPath+"/*.wav"),key=modelBuilder.findNumbers),dtype=str)
        else:
            audioPathList = None

        diagBlock = modelBuilder.DiagBlock(cuda=cuda,batchSize=batchSize,feat=visualFeat,pretrainDataSet=pretrainSet,audioFeat=audioFeat)

        diagBlock.detectDiagBlock(imagePathList,audioPathList,"test_exp",1)

def epochSiam(model,optim,log_interval,loader, epoch, args,writer,kwargs,mode):

    model.train()

    print("Epoch",epoch," : ",mode)

    total_loss = 0
    total_posDist = 0
    total_negDist = 0
    correct =0
    repList = None

    if not kwargs["audioModel"] is None:
        audioModel = kwargs["audioModel"]

    if kwargs["mining_mode"] == "offline":

        for batch_idx, (_,anch,pos,neg,anchAudio,posAudio,negAudio,_,_,_) in enumerate(loader):

            if (batch_idx % log_interval == 0):
                print("\t",batch_idx+1,"/",loader.batchNb)

            if args.cuda:
                anch,pos,neg = anch.cuda(),pos.cuda(),neg.cuda()

            anchRep = model(anch)
            posRep = model(pos)
            negRep = model(neg)

            if not kwargs["audioModel"] is None:
                anchAudRep = audioModel(anchAudio)
                posAudRep = audioModel(posAudio)
                negAudRep = audioModel(negAudio)

                if args.cuda:
                    anchAudRep,posAudRep,negAudRep = anchAudRep.cuda(),posAudRep.cuda(),negAudRep.cuda()

                anchRep = torch.cat((anchRep,anchAudRep),dim=1)
                posRep = torch.cat((posRep,posAudRep),dim=1)
                negRep = torch.cat((negRep,negAudRep),dim=1)

            #Loss
            p = kwargs["dist_order"]
            margin = kwargs["margin"]
            loss = torch.nn.functional.triplet_margin_loss(anchRep, posRep, negRep,margin=margin,p=p)

            posDist = torch.pow(torch.pow(anchRep-posRep,p).sum(dim=1),1/p)
            negDist = torch.pow(torch.pow(anchRep-negRep,p).sum(dim=1),1/p)

            correct += (posDist < negDist-margin).sum().float()/posDist.size(0)

            total_posDist += posDist.mean()
            total_negDist += negDist.mean()

            if mode == "train":
                loss.backward()
                optim.step()
                optim.zero_grad()
            else:
                loss.backward()
                optim.zero_grad()

            #Metrics
            total_loss += loss

            #if batch_idx == 3:
            #    break

    if mode == "train":
        torch.save(model.state_dict(), "../models/{}/{}_siam_epoch{}".format(args.exp_id,args.model_id, epoch))

    writeSummariesSiam(total_loss,correct,total_posDist,total_negDist,batch_idx+1,writer,epoch,mode,args.model_id)

def compDistMat(feat):

    featCol = feat.unsqueeze(1)
    featRow = feat.unsqueeze(0)

    featCol = featCol.expand(feat.size(0),feat.size(0),feat.size(1))
    featRow = featRow.expand(feat.size(0),feat.size(0),feat.size(1))

    simMatrix = torch.pow(featCol-featRow,2).sum(dim=2)

def epochSeq(model,optim,log_interval,loader, epoch, args,writer,kwargsTrain,mode):


    model.train()
    if args.train_visual:
        model.featModel.train()
    else:
        model.featModel.eval()

    print("Epoch",epoch," : ",mode)

    total_loss = 0
    total_cover = 0
    total_overflow = 0
    validBatch = 0

    for batch_idx, (data, audio,target,_) in enumerate(loader):

        if target.sum() > 0:
            if (batch_idx % log_interval == 0):
                print("\t",loader.sumL+1,"/",loader.nbShots)

            if args.cuda:
                data,audio, target = data.cuda(), audio.cuda(),target.cuda()
            output = model(data,audio)

            #Loss
            output_resh = output.view(output.size(0)*output.size(1),output.size(2))
            target_resh = target.view(target.size(0)*target.size(1))
            weights = getWeights(target,args.class_weight)
            #print(weights)
            #print(data.size(1))
            #print(F.softmax(output,dim=-1))
            #print(torch.argmax(F.softmax(output,dim=-1),dim=-1))
            #print(target)
            #print(target_resh)
            loss = F.cross_entropy(output_resh, target_resh.long(),weight=weights)
            #loss = torch.pow(F.softmax(output,dim=-1)[:,:,1],2).sum()
            if mode == "train":
                loss.backward()
                optim.step()
                optim.zero_grad()
            else:
                loss.backward()
                optim.zero_grad()

            #Metrics
            pred = torch.argmax(F.softmax(output,dim=-1),dim=-1)
            cov,overflow = processResults.binaryToMetrics(pred,target)
            total_cover += cov
            total_overflow += overflow
            total_loss += loss
            validBatch += 1

        #if batch_idx == 3:
        #    break

    if mode =="train":
        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id, epoch))

    writeSummaries(total_loss,total_cover,total_overflow,validBatch,writer,epoch,mode,args.model_id)

def getWeights(target,classWeight):

    oneNb = target.sum()
    zeroNb = target.numel() - oneNb

    unBalWeights = torch.Tensor([0.5,0.5])
    balWeights = torch.Tensor([oneNb/target.numel(),zeroNb/target.numel()])

    weights = unBalWeights*(1-classWeight)+balWeights*classWeight

    if target.is_cuda:
        weights = weights.cuda()

    return weights

def writeSummariesSiam(total_loss,correct,total_posDist,total_negDist,batchNb,writer,epoch,mode,model_id):

    total_loss /= batchNb
    total_posDist /= batchNb
    total_negDist /= batchNb
    accuracy = correct/batchNb

    writer.add_scalars('Losses',{model_id+"_"+mode:total_loss},epoch)
    writer.add_scalars('Accuracies',{model_id+"_"+mode:accuracy},epoch)
    writer.add_scalars('Pos_dist',{model_id+"_"+mode:total_posDist},epoch)
    writer.add_scalars('Neg_dist',{model_id+"_"+mode:total_negDist},epoch)

    if not os.path.exists("../results/{}_siam_epoch{}_metrics_{}.csv".format(model_id,epoch,mode)):
        header = "epoch,loss,accuracy,posDist,negDist"
    else:
        header = ""

    with open("../results/{}_siam_epoch{}_metrics_{}.csv".format(model_id,epoch,mode),"a") as text_file:
        print(header,file=text_file)
        print("{},{},{},{},{}".format(epoch,total_loss,accuracy,total_posDist,total_negDist),file=text_file)

def writeSummaries(total_loss,total_cover,total_overflow,validBatch,writer,epoch,mode,model_id):

    total_loss /= validBatch
    total_cover /= validBatch
    total_overflow /= validBatch
    f_score = 2*total_cover*(1-total_overflow)/(total_cover+1-total_overflow)

    writer.add_scalars('Losses',{model_id+"_"+mode:total_loss},epoch)
    writer.add_scalars('Coverages',{model_id+"_"+mode:total_cover},epoch)
    writer.add_scalars('Overflows',{model_id+"_"+mode:total_overflow},epoch)
    writer.add_scalars('F-scores',{model_id+"_"+mode:f_score},epoch)

    if not os.path.exists("../results/model{}_epoch{}_metrics_{}.csv".format(model_id,epoch,mode)):
        header = "epoch,loss,coverage,overflow,f-score"
    else:
        header = ""

    with open("../results/model{}_epoch{}_metrics_{}.csv".format(model_id,epoch,mode),"a") as text_file:
        print(header,file=text_file)
        print("{},{},{},{},{}\n".format(epoch,total_loss,total_cover,total_overflow,f_score),file=text_file)

def get_OptimConstructor_And_Kwargs(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
        optimConst = getattr(torch.optim,optimStr)
        if optimStr == "SGD":
            kwargs= {'momentum': momentum}
        elif optimStr == "Adam":
            kwargs = {}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'amsgrad':True}

    print("Optim is :",optimConst)

    return optimConst,kwargs

def findLastNumbers(weightFileName):
    '''Extract the epoch number of a weith file name.

    Extract the epoch number in a weight file which name will be like : "clustDetectNet2_epoch45".
    If this string if fed in this function, it will return the integer 45.

    Args:
        weightFileName (string): the weight file name
    Returns: the epoch number

    '''

    i=0
    res = ""
    allSeqFound = False
    while i<len(weightFileName) and not allSeqFound:
        if not weightFileName[len(weightFileName)-i-1].isdigit():
            allSeqFound = True
        else:
            res += weightFileName[len(weightFileName)-i-1]
        i+=1

    res = res[::-1]

    return int(res)

def initialize_Net_And_EpochNumber(net,exp_id,model_id,cuda,start_mode,init_path):
    '''Initialize a network

    If init is None, the network will be left unmodified. Its initial parameters will be saved.

    Args:
        net (CNN_RNN): the net to be initialised

        exp_id (string): the name of the experience
        model_id (int): the id of the network
        cuda (bool): whether to use cuda or not
        start_mode (str): a string indicating the start mode. Can be \'scratch\' or \'fine_tune\'.
        init_path (str): the path to the weight file of the net to use to initialise. Ignored is start_mode is \'scratch\'.

    Returns: the start epoch number
    '''

    if start_mode == "scratch":
        #Saving initial parameters
        torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id,model_id))
        startEpoch = 1
    elif start_mode == "fine_tune":

        params = torch.load(init_path)
        net.load_state_dict(params)
        startEpoch = findLastNumbers(init_path)

    return startEpoch

def evalAllImages(exp_id,model_id,model,audioModel,dataset_test,testLoader,cuda):

    for batch_idx, (data,audio, _,_,imagePaths) in enumerate(testLoader):

        imagePaths = imagePaths[0]
        videoName = imagePaths[0].split("/")[-3]
        print(videoName,"\t",batch_idx+1,"/",testLoader.batchNb)

        if not data is None:
            if cuda:
                data = data.cuda()
            data = data[:,:len(imagePaths)]
            data = data.view(data.size(0)*data.size(1),data.size(2),data.size(3),data.size(4))

        if not audio is None:
            if cuda:
                audio = audio.cuda()
            audio = audio[:,:len(imagePaths)]
            audio = audio.view(audio.size(0)*audio.size(1),audio.size(2),audio.size(3),audio.size(4))

        if not os.path.exists("../results/{}/{}".format(exp_id,videoName)):
            os.makedirs("../results/{}/{}".format(exp_id,videoName))

        if (not audioModel is None) and (not model is None):
            audioFeats = audioModel(audio)
            feats = model(data)
            feats = torch.cat((feats,audioFeats),dim=-1)
        elif (not audioModel is None):
            audioFeats = audioModel(audio)
            feats = audioFeats
        elif (not model is None):
            feats = model(data)
        for i,feat in enumerate(feats):
            imageName = os.path.basename(imagePaths[i])
            if not os.path.exists("../results/{}/{}/{}_{}.csv".format(exp_id,videoName,imageName,model_id)):

                np.savetxt("../results/{}/{}/{}_{}.csv".format(exp_id,videoName,imageName,model_id),feat.detach().cpu().numpy())

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--baseline', type=int,nargs=2, metavar='N',help='To run the baseline on every video of the dataset. The --dataset argument must\
                                                                        also be used. The values of the argument are the index of the first and the last video to process.')

    argreader.parser.add_argument('--comp_feat', action='store_true',help='To compute and write in a file the features of all images in the test set')
    argreader.parser.add_argument('--train_siam', action='store_true',help='To train a siamese network instead of a CNN-RNN')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id,args.model_id))

    writer = SummaryWriter("../results/{}".format(args.exp_id))

    if args.baseline:
        baselineAllVids(args.dataset_test,args.batch_size,args.feat,args.feat_audio,args.pretrain_dataset,args.cuda,args.baseline,args.audio_len)
    elif args.comp_feat:

        if args.feat != "None":
            featModel = modelBuilder.buildFeatModel(args.feat,args.pretrain_dataset,args.lay_feat_cut)
            if args.cuda:
                featModel = featModel.cuda()
            if args.init_path_visual != "None":
                featModel.load_state_dict(torch.load(args.init_path_visual))
            featModel.eval()
        else:
            featModel = None

        if args.feat_audio != "None":
            audioFeatModel = modelBuilder.buildAudioFeatModel(args.feat_audio)
            if args.cuda:
                audioFeatModel = audioFeatModel.cuda()
            if args.init_path_audio != "None":
                audioFeatModel.load_state_dict(torch.load(args.init_path_audio))
            audioFeatModel.eval()
        else:
            audioFeatModel = None

        img_size = (args.img_width,args.img_heigth)
        testLoader = load_data.TestLoader(args.val_l,args.dataset_test,args.test_part_beg,args.test_part_end,(args.img_width,args.img_heigth),args.audio_len)


        evalAllImages(args.exp_id,args.model_id,featModel,audioFeatModel,args.dataset_test,testLoader,args.cuda)

    else:

        if args.img_width == -1 or args.img_heigth == -1:
            img_size = None
        else:
            img_size = (args.img_width,args.img_heigth)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        if args.train_siam:

            net = modelBuilder.buildFeatModel(args.feat,args.pretrain_dataset,args.lay_feat_cut)

            if args.feat_audio != "None":
                audioNet = modelBuilder.buildAudioFeatModel(args.feat_audio)
                audioLen = args.audio_len
            else:
                audioNet = None
                audioLen = 0

            trainLoader = load_data.PairLoader(args.dataset_train,args.batch_size,img_size,args.train_part_beg,args.train_part_end,True,audioLen)
            valLoader = load_data.PairLoader(args.dataset_val,args.val_batch_size,img_size,args.val_part_beg,args.val_part_end,True,audioLen)

            trainFunc = epochSiam
            valFunc = epochSiam
            kwargs = {'margin':args.margin,"dist_order":args.dist_order,"mining_mode":args.mining_mode,"audioModel":audioNet}

        else:

            trainLoader = load_data.TrainLoader(args.batch_size,args.dataset_train,args.train_part_beg,args.train_part_end,args.l_min,args.l_max,(args.img_width,args.img_heigth),args.audio_len)
            valLoader = load_data.TestLoader(args.val_l,args.dataset_val,args.test_part_beg,args.test_part_end,(args.img_width,args.img_heigth),args.audio_len)
            #testLoader = load_data.TestLoader(args.dataset_test,args.batch_size,args.l_min,args.l_max,img_size,propStart=args.test_part_beg,propEnd=args.test_part_end,\
            #                                    shuffle=False,removeUselessSeq=True,visualFeat=args.feat != "None",audioFeat=args.feat_audio,audioLen=args.audio_len)

            #Building the net
            net = modelBuilder.netBuilder(args)

            trainFunc = epochSeq
            valFunc = epochSeq
            kwargs = {}

        img_size = (args.img_width,args.img_heigth)

        if args.cuda:
            net = net.cuda()
            if args.multi_gpu:
                net = DataParallel(net)

        #Getting the contructor and the kwargs for the choosen optimizer
        optimConst,kwargsOpti = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)

        startEpoch = initialize_Net_And_EpochNumber(net,args.exp_id,args.model_id,args.cuda,args.start_mode,args.init_path)

        #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
        #the args.lr argument will be a float and not a float list.
        #Converting it to a list with one element makes the rest of processing easier
        if type(args.lr) is float:
            args.lr = [args.lr]

        lrCounter = 0
        for epoch in range(startEpoch, args.epochs + 1):

            #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
            #The optimiser have to be rebuilt every time the learning rate is updated
            if (epoch-1) % ((args.epochs + 1)//len(args.lr)) == 0 or epoch==startEpoch:

                kwargsOpti['lr'] = args.lr[lrCounter]
                print("Learning rate : ",kwargsOpti['lr'])
                if args.train_visual:
                    optim = optimConst(net.parameters(), **kwargsOpti)
                else:
                    optim = optimConst(net.getParams(), **kwargsOpti)

                if lrCounter<len(args.lr)-1:
                    lrCounter += 1

            valFunc(net,optim,args.log_interval,valLoader,epoch,args,writer,kwargs,"val")
            trainFunc(net,optim,args.log_interval,trainLoader,epoch,args,writer,kwargs,"train")
            #val(net,optim,valLoader,epoch,args,writer,5)

if __name__ == "__main__":
    main()
