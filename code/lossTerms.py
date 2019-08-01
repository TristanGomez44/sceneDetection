import torch
from torch.nn import functional as F

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
