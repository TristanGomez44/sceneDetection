import torch
from torch.nn import functional as F

import matplotlib.pyplot as plt
plt.switch_backend('agg')


import sys

from argparse import Namespace

import numpy as np

def addDistTerm(loss,args,output,target):

    #Adding distance to closest target term
    if args.dist_weight > 0 and target.sum() > 0:

        inds = torch.arange(output.size(1)).unsqueeze(0).unsqueeze(2).to(target.device)
        distTerm = 0
        nonZeroEx = 0
        for i,targSeq in enumerate(target):
            if targSeq.sum() > 0:
                targInds = targSeq.nonzero().permute(1,0).unsqueeze(0)
                distTerm +=args.dist_weight*((output[i]>0.5).long()*torch.min(torch.abs(inds-targInds),dim=2)[0]).float().mean()
                nonZeroEx += 1

        loss += distTerm/nonZeroEx
    return loss
def addAdvTerm(loss,args,feat,featModel,discrModel,discrIter,discrOptim):
    #Adding adversarial term
    if args.adv_weight > 0:
        discrModel.zero_grad()

        frames,gt,_ = discrIter.__next__()
        frames,gt = frames.to(loss.device),gt.to(loss.device)

        featAdv = featModel(frames)
        featAdv = torch.cat((featAdv,feat.view(feat.size(0)*feat.size(1),-1)),dim=0)
        gt = torch.cat((gt,torch.zeros(feat.size(0)*feat.size(1)).unsqueeze(1).to(loss.device)),dim=0)

        pred = discrModel(featAdv)

        discMeanAcc = ((pred.data > 0.5).float() == gt.data).float().mean()

        dLoss = args.adv_weight*F.binary_cross_entropy(pred,gt)
        dLoss.backward(retain_graph=True)
        discrOptim.step()

        loss += -args.adv_weight*F.binary_cross_entropy(pred,gt)
    else:
        discMeanAcc = 0

    return loss,discMeanAcc
def addSiamTerm(loss,args,featBatch,target):
    target = torch.cumsum(target,dim=-1)

    distPos,distNeg = 0,0

    if args.siam_weight > 0:
        #Parsing each example of the batch
        for i,feat in enumerate(featBatch):
            inds1,inds2 = torch.randint(feat.size(0),size=(2,args.siam_nb_samples))
            feat1,feat2 = feat[inds1],feat[inds2]
            targ1,targ2 = target[i][inds1],target[i][inds2]

            dist = torch.pow(feat1-feat2,2).sum(dim=-1)
            targ = (targ1==targ2).float()

            loss += args.siam_weight*(targ*dist+(1-targ)*F.relu(args.siam_margin-dist)).mean()

            distPos += dist[targ.long()].mean()
            distNeg += dist[1-targ.long()].mean()

    return loss,distPos/len(featBatch),distNeg/len(featBatch)


def plotSig(signal,name,iter):
    if iter%10000==0:
        for i in range(len(signal)):
            plt.figure(figsize=(15,8))

            if (signal[i] < 0).sum() > 0:
                plt.ylim(-1,1)
            else:
                plt.ylim(0,1)
            plt.plot(signal[i],marker="*")
            plt.savefig("../vis/{}{}_iter{}.png".format(name,i,iter))
            plt.close()

def integrate(signal,name="output",iter=0,plot=False):

    signal_cs = torch.cumsum(signal, dim=-1)
    signal_int_pair = (signal_cs % 1)*(signal_cs.int() % 2 == 0)
    signal_int_impair = (1-(signal_cs % 1))*(signal_cs.int() % 2 == 1)
    signal_int = signal_int_pair + signal_int_impair

    if plot:
        plotSig(signal_int.cpu().detach().numpy(),"{}_int".format(name),iter=iter)
        plotSig(signal.cpu().detach().numpy(),name,iter=iter)

    return signal_int

def addIoUTerm(loss,args,output,target,iter=0,plot=False):

    if args.iou_weight > 0:

        #output = output*2-1
        #output_int = integrate(output,name="output",iter=iter,plot=plot)
        output_int = output
        target_int = integrate(target,name="target",iter=iter,plot=plot)

        output_int = output_int
        target_int = target_int

        if plot:
            iouTime = (output_int*target_int)/torch.clamp(target_int+output_int,0,1)
            plotSig(iouTime.cpu().detach().numpy(),"iou",iter=iter)

        rev_output_int = 1 - output_int
        rev_target_int = 1 - target_int
        iou = ((output_int*target_int)/torch.clamp(target_int+output_int,0,1)).mean()
        rev_iou = ((rev_output_int*rev_target_int)/torch.clamp(rev_target_int+rev_output_int,0.001,1)).mean()

        if plot:
            revIouTime = (rev_output_int*rev_target_int)/torch.clamp(rev_target_int+rev_output_int,0,1)
            plotSig(revIouTime.cpu().detach().numpy(),"rev_iou",iter=iter)
            plotSig((revIouTime+iouTime).cpu().detach().numpy(),"iou_sum",iter=iter)

        loss += args.iou_weight*(iou+rev_iou)

    return loss

if __name__ == "__main__":


    loss = 0
    args = Namespace(iou_weight=1)
    #output = torch.zeros(1,20)
    #output[0] = torch.tensor([0.5,0.1,0.4,0.3,0.4,0.7,0.8,0.1,0.05,0.1])
    #output[0] = 0.25
    output_param = torch.normal(0,0.125,(1,80),requires_grad=True)
    output = torch.sigmoid(output_param)
    target = torch.zeros(1,80)
    tens = torch.tensor([0 ,1  ,0   ,0  ,0  ,1  ,0  ,0  ,0   ,0  ,0 ,1  ,0   ,0  ,0  ,1  ,0  ,0  ,0   ,0  ,0 ,1  ,0   ,0  ,0  ,1  ,0  ,0  ,0   ,0  ,0 ,1  ,0   ,0  ,0  ,1  ,0  ,0  ,0   ,0  ])
    target[0] = torch.cat((tens,tens),dim=0)

    opti = torch.optim.SGD([output_param],lr=0.05)

    for i in range(100000):
        if i%10000 == 0:
            print("iter",i)

        output = torch.sigmoid(output_param)

        loss = 0
        loss = -addIoUTerm(loss,args,output,target,i,plot=True)
        loss.backward()

        opti.step()
        opti.zero_grad()
