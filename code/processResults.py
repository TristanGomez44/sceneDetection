
from args import ArgReader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
def main(argv=None):


    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--plot_scenebounds',type=str,metavar='RESFILE',help='To plot the scene boundaries found by a model in an experiment. The argument value is the id of the model. The --dataset argument\
                                    must also be set.')

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_scenebounds:


        sceneCutsPathList = glob.glob("../results/{}/*_{}.csv".format(args.exp_id,args.plot_scenebounds))
        for sceneCutsPath in sceneCutsPathList:

            sceneCuts = np.genfromtxt(sceneCutsPath)

            vidName = "_".join(os.path.basename(sceneCutsPath).split("_")[:-1])
            vidFold = "/".join(os.path.dirname(sceneCutsPath).split("/")[:-1])

            simMat = np.genfromtxt(vidFold+"/"+vidName+".csv")

            plt.figure()
            plt.imshow(simMat.astype(int), cmap='gray', interpolation='nearest')

            for sceneCut in sceneCuts:
                plt.plot([sceneCut-10,sceneCut+10],[sceneCut+10,sceneCut-10],"-",color="red")

            plt.xlim(0,len(simMat))
            plt.ylim(len(simMat),0)
            plt.savefig("../vis/{}/{}_{}.png".format(args.exp_id,vidName,args.plot_scenebounds))

if __name__ == "__main__":
    main()
