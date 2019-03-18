import sys
import inceptionFeat
from torchvision.models.inception import inception_v3
import cv2
import glob
import torch
from skimage.transform import resize
import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
import time

class DiagBlock():

    def __init__(self,cuda=True,batchSize=32):
        self.incep = None

        self.cuda = cuda
        self.batchSize = batchSize


    def detectDiagBlock(self,imagePathList,sceneNb=5):

        foldName = os.path.dirname(imagePathList[0]).split("/")[-2]
        print(foldName)

        if not os.path.exists("../results/{}.csv".format(foldName)):

            if self.incep is None:
                self.incep = inceptionFeat.inception_v3(pretrained=True)
                if self.cuda:
                    self.incep = self.incep.cuda()
                self.incep.eval()

            def preprocc(x):
                x = cv2.imread(x)
                x = resize(x, (299, 299,3), anti_aliasing=True,mode="constant")

                if self.cuda:
                    x = torch.tensor(x).cuda()
                else:
                    x = torch.tensor(x)

                x = x.permute(2,0,1).float().unsqueeze(0)

                return x

            print("Computing features")

            #The modulo term increases the total number of batch by one in case
            #One last small batch is required
            nbBatch = len(imagePathList)//self.batchSize + (len(imagePathList) % self.batchSize == 0)
            feat = None
            for i in range(nbBatch):

                indMax = min((i+1)*self.batchSize,len(imagePathList))

                imageBatch = list(map(preprocc,imagePathList[i*self.batchSize:indMax]))

                imageBatch = torch.cat(imageBatch)

                featBatch = self.incep(imageBatch).detach()

                if feat is None:
                    feat = featBatch
                else:
                    feat = torch.cat([feat,featBatch])

            print("Computing the similarity matrix")

            featCol = feat.unsqueeze(1)
            featRow = feat.unsqueeze(0)

            featCol = featCol.expand(feat.size(0),feat.size(0),feat.size(1))
            featRow = featRow.expand(feat.size(0),feat.size(0),feat.size(1))

            simMatrix = torch.pow(featCol-featRow,2).sum(dim=2)

            simMatrix_np = simMatrix.detach().cpu().numpy()
            np.savetxt("../results/{}.csv".format(foldName),simMatrix_np)

            plt.imshow(simMatrix_np.astype(int), cmap='gray', interpolation='nearest')
            plt.savefig("../vis/{}.png".format(foldName))

        else:
            print("Reading similarity matrix")

            simMatrix = torch.tensor(np.genfromtxt("../results/{}.csv".format(foldName)))

        print("Computing scene number")

        s = np.linalg.svd(simMatrix,compute_uv=False)
        s = np.log(s[np.where(s > 1)])

        H = np.array([len(s)-1,s[0]-s[len(s)-1]])
        I = np.concatenate((s[:,np.newaxis],np.arange(len(s))[:,np.newaxis]),axis=1)

        K = np.argmin((I*H).sum(axis=1))+1

        print("Detecting {} scenes".format(K))

        start = time.time()

        N = len(imagePathList)
        K = sceneNb
        pMax = N*N

        C = torch.zeros((N,K,pMax))
        I = torch.zeros((N,K,pMax)).int()
        P = torch.zeros((N,K,pMax))
        G = torch.zeros(N)

        if self.cuda:
            print("Putting tensors on cuda")
            simMatrix = simMatrix.cuda()
            C,I,P,G = C.cuda(),I.cuda(),P.cuda(),G.cuda()

        if not os.path.exists("../results/B_N{}_K{}.csv".format(N,K)):

            #Indicates which value of p are possible
            B = torch.zeros((N,K,pMax)).byte()
            lineEnum = torch.arange(N-1,-1,-1).unsqueeze(1).expand(N,pMax)
            colEnum = torch.arange(N*N-1,-1,-1).unsqueeze(0).expand(N,pMax)

            tens1 = torch.arange(N).unsqueeze(1).expand(N,pMax)
            tens2 = torch.pow(torch.arange(pMax),2).unsqueeze(0).expand(N,pMax)

            if self.cuda:
                B = B.cuda()
                tens1,tens2, = tens1.cuda(),tens2.cuda()
                lineEnum = lineEnum.cuda()
                colEnum = colEnum.cuda()

            B[:,0,:] = (tens1==tens2)

            for k in range(1,K):

                precMat = B[:,k-1,:]
                print(k)

                for n in range(1,N+1):
                    print("\t",n)

                    """
                    print("Computing plausible values")
                    p_plausible_values = (np.ceil(n*n/k) < arr1) * (arr1 < (n-k+1)*(n-k+1)+k)
                    print("Computing values to select")
                    p_values_to_select = p_plausible_values * p_values_to_sum

                    print("Puting the tensor on cuda")
                    if self.cuda:
                        tens2 = colEnum[:,N-n,:].cuda()
                    else:
                        tens2 = colEnum[:,N-n,:].long()

                    print("Actually selecting them")
                    tens2 = tens2*p_values_to_select.long()

                    #print(p_plausible_values.sum())
                    #print(tens2[p_plausible_values,N-n,:].size())

                    #tens2[:,N-n,:][p_values_to_select] = 0
                    """

                    for p in range(int(np.ceil(n*n/k)),(n-k+1)*(n-k+1)+k):

                        tens1 = lineEnum[N-n:,pMax-p:]
                        tens2 = colEnum[N-n:,pMax-p:]

                        B[n-1,k,p-1] = precMat[:n,:p][(tens1==tens2)].any()

            np.savetxt(B.view(B.size(0),-1).detach().cpu().numpy(),"../results/B_N{}_K{}.csv".format(N,K))

        possibleValues = torch.nonzero(B)
        print(time.time()-start)
        print(len(possibleValues),possibleValues,B.numel())
        sys.exit(0)

        for n,k,p in possibleValues:

        #for k in range(1,K+1):
        #    for n in range(N):
        #        for p in range(pMax):
            if k == 0:
                C[n,0,p] = simMatrix[n:,n:].sum()/(p+(N-n+1)*(N-n+1)-N)
                I[n,0,p] = N
                P[n,0,p] = (N-n+1)*(N-n+1)
            else:

                i=n+1
                while i<N-1 and p+(i-n+1)*(i-n+1)<pMax:
                    a = p+(i-n+1)*(i-n+1)
                    G[i] = simMatrix[n:i+1,n:i+1].sum()/(a+P[i+1,k-1,a]-N)
                    i+=1

                minimum,argmin = torch.min(G[n:i-1],dim=0)

                C[n,k,p] = minimum
                I[n,k,p] = argmin

                b = (I[n,k,p]-n+1)*(I[n,k,p]-n+1)

                P[n,k,p] = b+P[I[n,k,p]+1,k-1,p+b]

        P_tot = 0
        sceneSplit = [0]
        for i in range(1,K+1):
            sceneSplit.append(I[sceneSplit[i-1]+1,K-i,P_tot])
            P_tot += (sceneSplit[i]-sceneSplit[i-1])*(sceneSplit[i]-sceneSplit[i-1])

        print("Sanity check :",P_tot,P[1,K-1,0])

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))


def main():

    imagePathList = np.array(sorted(glob.glob("../data/big_buck_bunny_480p_surround-fix/middleFrames/*"),key=findNumbers),dtype=str)

    diagBlock = DiagBlock(cuda=True)

    diagBlock.detectDiagBlock(imagePathList)
if __name__ == "__main__":
    main()
