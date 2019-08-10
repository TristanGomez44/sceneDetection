import requests
import json
import torch
from args import ArgReader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
import argparse
import re
import time
import load_data
import metrics
import matplotlib.cm as cm

def main(argv=None):

    parser = argparse.ArgumentParser(description='Plot the performance of a model that just output a random segmentation \
                                    and compare it to model that takes the images into account.')

    parser.add_argument('--dataset', metavar='DATASET',help='The dataset to process',type=str)
    parser.add_argument('--model_label_list', metavar='LABEL',help='The list of label for the models to compare with ',type=str,nargs="*")
    parser.add_argument('--model_perf_list', metavar='LABEL',help='The list of metric values for each of the models to compare with ',type=float,nargs="*")
    parser.add_argument('--metric', metavar='METRIC',help='The metric to use.',type=str)
    parser.add_argument('--nb_trial', metavar='NB',help='The number of sampling of the random model for each threshold and each video',type=int,default=10)

    args = parser.parse_args()

    videoPaths = load_data.findVideos(args.dataset,propStart=0,propEnd=1)
    probs = torch.arange(10).float()/10

    perfArr = np.zeros((len(probs),len(videoPaths),args.nb_trial))

    for i,prob in enumerate(probs):
        print(i,prob)
        distr = torch.distributions.bernoulli.Bernoulli(probs=prob)

        for j,videoPath in enumerate(videoPaths):
            vidName = os.path.basename(os.path.splitext(videoPath)[0])
            nbShots = len(np.genfromtxt(os.path.splitext(videoPath)[0]+"/result.csv"))
            gt = load_data.getGT(args.dataset,vidName).astype(int)

            for k in range(args.nb_trial):
                randPred = distr.sample((nbShots,)).int()
                metr_dict = metrics.binaryToAllMetrics(randPred[np.newaxis,:],torch.tensor(gt[np.newaxis,:]),lenPond=True)
                perfArr[i,j,k] = metr_dict[args.metric]

    perfArr = perfArr.mean(axis=1)

    cmap = cm.rainbow(np.linspace(0, 1, len(args.model_label_list)))

    plt.figure()
    plt.ylim(0,1)
    plt.xlabel("Probability of scene change")
    plt.ylabel(args.metric)
    plt.errorbar(np.array(probs),perfArr.mean(axis=1),yerr=1.96*perfArr.std(axis=1)/np.sqrt(args.nb_trial))
    for i in range(len(args.model_label_list)):
        plt.hlines(args.model_perf_list[i],0,1,label=args.model_label_list[i],color=cmap[i])

    plt.legend(loc="lower right")
    plt.savefig("../vis/{}_{}_randomModel.png".format(args.dataset,args.metric))

if __name__ == "__main__":
    main()
