import utils
import numpy as np
import torch
import scipy as sp

def binaryToMetrics(pred,target):
    ''' Computes metrics of a predicted scene segmentation using a gt and a prediction encoded in binary format

    Args:
    - pred (list): the predicted scene segmentation. It is a list indicating for each shot if it is the begining of a new scene or not. A 1 indicates that \
                    the shot is the first shot of a new scene.
    - target (list): the ground truth scene segmentation. Formated the same way as pred.

    '''

    predBounds = []
    targBounds = []

    for i in range(len(pred)):

        pred_bounds = utils.binaryToSceneBounds(pred[i])
        targ_bounds = utils.binaryToSceneBounds(target[i])

        if len(targ_bounds) > 1:
            predBounds.append(pred_bounds)
            targBounds.append(targ_bounds)

    cov_val,overflow_val,iou_val = 0,0,0
    for pred,targ in zip(predBounds,targBounds):

        cov_val += coverage(np.array(targ),np.array(pred))
        overflow_val += overflow(np.array(targ),np.array(pred))
        iou_val += IoU(np.array(targ),np.array(pred))

    cov_val /= len(targBounds)
    overflow_val /= len(targBounds)
    iou_val /= len(targBounds)

    return cov_val,overflow_val,iou_val

def binaryToAllMetrics(predBin,targetBin):
    ''' Computes the IoU of a predicted scene segmentation using a gt and a prediction encoded in binary format

    This computes IoU relative to prediction and to ground truth and also computes the mean of the two. \

    Args:
    - predBin (list): the predicted scene segmentation. It is a list indicating for each shot if it is the begining of a new scene or not. A 1 indicates that \
                    the shot is the first shot of a new scene.
    - targetBin (list): the ground truth scene segmentation. Formated the same way as pred.


    '''

    predBounds = []
    targBounds = []

    for i in range(len(predBin)):

        pred_bounds = utils.binaryToSceneBounds(predBin[i])
        targ_bounds = utils.binaryToSceneBounds(targetBin[i])

        if len(targ_bounds) > 1:
            predBounds.append(pred_bounds)
            targBounds.append(targ_bounds)

    iou,iou_pred,iou_gt,over,cover,ded = 0,0,0,0,0,0
    for i,(pred,targ) in enumerate(zip(predBounds,targBounds)):

        iou_pred += IoU_oneRef(np.array(targ),np.array(pred))
        iou_gt += IoU_oneRef(np.array(pred),np.array(targ))
        iou += iou_pred*0.5+iou_gt*0.5
        over += overflow(np.array(targ),np.array(pred))
        cover += coverage(np.array(targ),np.array(pred))
        ded += computeDED(targetBin[i].unsqueeze(0),predBin[i].unsqueeze(0))

    iou_pred /= len(targBounds)
    iou_gt /= len(targBounds)
    iou /= len(targBounds)
    over /= len(targBounds)
    cover /= len(targBounds)
    ded /= len(targBounds)

    f_score = 2*cover*(1-over)/(cover+1-over)

    return {"IoU":iou,"IoU_pred":iou_pred,"IoU_gt":iou_gt,"F-score":f_score,"DED":ded}

def coverage(gt,pred):
    ''' Computes the coverage of a scene segmentation

    Args:
    - gt (array): the ground truth segmentation. It is a list of gr interval (each interval is a tuple made of the first and the last shot index of the scene)
    - pred (array): the predicted segmentation. Same format as gt

    Returns:
    - the mean coverage of the predicted scene segmentation

    '''

    cov_gt_array = np.zeros(len(gt))
    for i,scene in enumerate(gt):

        cov_pred_array = np.zeros(len(pred))
        for j,scene_pred in enumerate(pred):

            cov_pred_array[j] = inter(scene,scene_pred)/leng(scene)
        cov_gt_array[i] = cov_pred_array.max()
        #cov_gt_array[i] *= leng(scene)/(gt[-1,1]+1)

    #print(cov_gt_array)
    #return cov_gt_array.sum()
    return cov_gt_array.mean()

def leng(scene):
    ''' The number of shot in an interval, i.e. a scene '''

    return scene[1]-scene[0]+1

def overflow(gt,pred):
    ''' Computes the overflow of a scene segmentation

    Args:
    - gt (array): the ground truth segmentation. It is a list of gr interval (each interval is a tuple made of the first and the last shot index of the scene)
    - pred (array): the predicted segmentation. Same format as gt

    Returns:
    - the mean overflow of the predicted scene segmentation

    '''

    ov_gt_array = np.zeros(len(gt))
    for i,scene in enumerate(gt):



        #if scene[0] == 3 and scene[1] == 21:
        #    print(scene)
        #print("Ground truth scene :",i)
        #print(scene)
        ov_pred_array = np.zeros(len(pred))
        for j,scene_pred in enumerate(pred):
            #if scene[0] == 3 and scene[1] == 21:
            #    print(scene_pred,minus(scene_pred,scene),(inter(scene_pred,gt[i-1])>0),(inter(scene_pred,gt[i+1])>0))
            #ov_pred_array[j] = minus(scene_pred,scene)*(inter(scene_pred,gt[i-1])>0)*(inter(scene_pred,gt[i+1])>0)
            #ov_pred_array[j] = minus(scene_pred,scene)*(inter(scene_pred,gt[i-1])+inter(scene_pred,gt[i+1])>0)
            ov_pred_array[j] = minus(scene_pred,scene)*min(1,inter(scene_pred,scene))

        #if scene[0] == 3 and scene[1] == 21:
        #    print("ov_pred_array",ov_pred_array,(leng(gt[i-1])+leng(gt[i+1])))
        if i>0 and i<len(gt)-1:
            ov_gt_array[i] = min(ov_pred_array.sum()/(leng(gt[i-1])+leng(gt[i+1])),1)
        elif i == 0:
            ov_gt_array[i] = min(ov_pred_array.sum()/(leng(gt[i+1])),1)
        elif i == len(gt)-1:

            ov_gt_array[i] = min(ov_pred_array.sum()/(leng(gt[i-1])),1)

        #ov_gt_array[i] *= leng(scene)/(gt[-1,1]+1)

        #print(leng(scene),gt[-1,1],leng(gt[0]),leng(gt[-1]))

    #print(ov_gt_array)
    #return ov_gt_array.sum()
    return ov_gt_array.mean()

def IoU(gt,pred):
    ''' Computes the Intersection over Union of a scene segmentation

    Args:
    - gt (array): the ground truth segmentation. It is a list of gr interval (each interval is a tuple made of the first and the last shot index of the scene)
    - pred (array): the predicted segmentation. Same format as gt

    Returns:
    - the IoU of the predicted scene segmentation with the ground-truth

    '''

    #The IoU is first computed relative to the ground truth and then relative to the prediction
    return 0.5*(IoU_oneRef(gt,pred)+IoU_oneRef(pred,gt))

def IoU_oneRef(sceneCuts1,sceneCuts2):
    ''' Compute the IoU of a segmentation relative another '''

    #Will store the IoU of every scene from sceneCuts1 with every scene from sceneCuts2
    iou = np.zeros((len(sceneCuts1),len(sceneCuts2),2))

    iou_mean = 0
    for i in range(len(sceneCuts1)):

        iou = np.zeros(len(sceneCuts2))
        for j in range(len(sceneCuts2)):
            iou[j] = inter(sceneCuts1[i],sceneCuts2[j])/union(sceneCuts1[i],sceneCuts2[j])

        iou_mean += iou.max()
    iou_mean /= len(sceneCuts1)

    return iou_mean

def union(a,b):
    ''' The union between two intervals '''

    return b[1]-b[0]+1+a[1]-a[0]+1-inter(a,b)

def inter(a,b):
    ''' The intersection between two intervals '''

    if b[0] > a[1] or a[0] > b[1]:
        return 0
    else:
        return min(a[1],b[1])-max(a[0],b[0])+1

def minus(a,b):
    ''' the interval a minus the interval b '''

    totalLen = 0
    bVal = np.arange(int(b[0]),int(b[1])+1)

    for shotInd in range(int(a[0]),int(a[1])+1):
        if not shotInd in bVal:
            totalLen += 1

    return totalLen

def computeDED(segmA,segmB):
    """ Computes the differential edit distance.

    Args:
        - segmA (array) a scene segmentation in the binary format. There is one binary digit per shot.\
         1 if the shot starts a new scene. 0 else.
        - segmB (array) another scene segmentation in the same format as segmA.
     """
    segmA,segmB = torch.cumsum(segmA,dim=-1),torch.cumsum(segmB,dim=-1)

    ded = 0

    #For each example in the batch
    for i in range(len(segmA)):

        #It is required that segmA is the sequence with the greatest number of scenes
        if segmB[i].max() > segmA[i].max():
            segmA[i],segmB[i] = segmB[i],segmA[i]
        else:
            segmA[i],segmB[i] = segmA[i],segmB[i]
        occMat = torch.zeros((torch.max(segmB[i])+1,torch.max(segmA[i])+1))
        for j in range(len(segmA[i])):
            occMat[segmB[i][j],segmA[i][j]] += 1

        costMat = torch.max(occMat)-occMat

        assign = sp.optimize.linear_sum_assignment(costMat)

        correctAssignedShots = np.array([occMat[p[0],p[1]] for p in zip(assign[0],assign[1])]).sum()

        ded += (len(segmB[i])-correctAssignedShots)/len(segmB[i])

    return ded/len(segmA)
