#include <torch/torch.h>
#include <iostream>
#include <string>
#include <stdlib.h>     /* strtod */
#include <vector>
#include <fstream>

int main(int argc, char *argv[]){



  std::string matrixPath = argv[1];
  std::string outPath = argv[2];
  int N = atoi(argv[3]);
  int K = atoi(argv[4]);
  int pMax = N*N;
  std::string device = argv[5];

  torch::Tensor simMat = torch::zeros({N,N},device);

  std::ifstream f(matrixPath);
  float elem;
  for (int i=0;i<N;++i){
    for (int j=0;j<N;++j){
        f >> elem;
        simMat[i][j] = elem;
    }
  }

  N = N;
  pMax = pMax;
  simMat = simMat.slice(0,0,N).slice(1,0,N);

  torch::Tensor C = torch::ones({N,K,pMax},device)*std::numeric_limits<double>::quiet_NaN();
  torch::Tensor I = torch::ones({N,K,pMax},device)*std::numeric_limits<double>::quiet_NaN();
  torch::Tensor P = torch::ones({N,K,pMax},device)*std::numeric_limits<double>::quiet_NaN();
  torch::Tensor G = torch::ones({N-1},device)*std::numeric_limits<double>::quiet_NaN();

  torch::Tensor nValues = torch::arange(0,N);
  torch::Tensor pValues = torch::arange(0,pMax);

  if(device=="cuda"){
    nValues = nValues.cuda();
    pValues = pValues.cuda();
  }

  torch::Tensor simMat_exp,sumVec;

  torch::Tensor ind0,ind1,ind2,indSumAreas;

  torch::Tensor mask;

  torch::Tensor summedAreas;

  torch::Tensor P_k_sel,C_k_sel;

  torch::Tensor G_paral;

  I.slice(0,0,N).slice(2,0,pMax).select(1,0) = N;

  for(int k=0;k<K;++k){
    std::cout << "k = " << k << "\n";
    for(int n=1;n<N+1;++n){
      std::cout << "\tn = " << n << "\n";

      if(k==0){

        P.select(0,n-1).select(0,0) = (N-n+1)*(N-n+1);
        C.select(0,n-1).select(0,0).slice(0,n-1,n*n) = simMat.slice(0,n,N).slice(1,n,N).sum()/(torch::arange(n-1,n*n)+(N-n+1)*(N-n+1)-N);

      }else{

        // Computation of the numerator
        //std::cout << "Ind init" << "\n";
        ind0 = nValues.slice(0,n-1,N-1).unsqueeze(1).unsqueeze(2).expand({N-n,N-n+1,N-n+1});
        ind1 = nValues.slice(0,n-1,N).unsqueeze(0).unsqueeze(2).expand({N-n,N-n+1,N-n+1});
        ind2 = nValues.slice(0,n-1,N).unsqueeze(0).unsqueeze(1).expand({N-n,N-n+1,N-n+1});

        //std::cout << "sim Mat expand and mask" << "\n";
        simMat_exp = simMat.slice(0,n-1,N).slice(1,n-1,N).unsqueeze(0).expand({N-n,N-n+1,N-n+1});
        simMat_exp = simMat_exp*((ind1 <= (ind0))*(ind2 <= (ind0))).to(simMat_exp.dtype());

        //std::cout << "sim mat sum" << "\n";
        sumVec = simMat_exp.sum(2).sum(1);

        //Computation of the denominator
        //std::cout << "Ind init" << "\n";
        ind0 = nValues.unsqueeze(1).expand({N,pMax});
        ind1 = pValues.unsqueeze(0).expand({N,pMax});

        //std::cout << "Slicing" << "\n";
        indSumAreas = nValues.slice(0,n,N);

        for(int p=n-1;p<n*n;++p){

          //std::cout << "n=" << n << "p=" << p << "\n";

          //std::cout << "Computing the mask" << "\n";
          mask = (ind1 == p+(ind0-n+1)*(ind0-n+1))*(ind0>(n-1));

          //std::cout << "Computing summed areas" << "\n";
          summedAreas = p+(indSumAreas-n+1)*(indSumAreas-n+1);

          //std::cout << "Slicing and selecting P and C" << "\n";
          P_k_sel = P.select(1,k-1).masked_select(mask);
          C_k_sel = C.select(1,k-1).masked_select(mask);

          //std::cout << "Computing G" << "\n";

          G_paral = (sumVec/(summedAreas+P_k_sel-N)+C_k_sel).slice(0,1,N-n+1);

          /*
          std::cout << sumVec << "\n";
          std::cout << summedAreas << "\n";
          std::cout << P_k_sel << "\n";
          std::cout << C_k_sel << "\n";
          std::cout << G_paral << "\n";

          std::cout << "G length :" << G_paral.size(0) << "\n";

          std::cout << "Computing the min" << "\n";
          */
          int argmin;
          float min;
          if (G_paral.size(0) == 0){
            min = 0;
            argmin=0;
          }else{
            //std::cout << "A real minimum" << "\n";
            min = G_paral.min().item<float>();
            argmin = G_paral.argmin().item<int>();
          }

          C.select(0,n-1).select(0,k).select(0,p) = min;
          I.select(0,n-1).select(0,k).select(0,p) = argmin+n-1;

          //std::cout << "Computing the area" << "\n";
          int b = (I.select(0,n-1).select(0,k).select(0,p)-n+1).item<int>();
          b = b*b;

          P.select(0,n-1).select(0,k).select(0,p) = b+P.select(0,I.select(0,n-1).select(0,k).select(0,p).item<int>())\
                                                       .select(0,k-1)\
                                                       .select(0,p+b);

          //if((n==11)*(k==4)){
          //  std::cout  <<   P.select(0,n-1).select(0,k).select(0,p) << "\n";
          //  std::cout  <<   C.select(0,n-1).select(0,k).select(0,p) << "\n";
          //  std::cout  <<   I.select(0,n-1).select(0,k).select(0,p) << "\n";
          //  return 0;
          //}

        }
      }
    }
  }

  int P_tot=0;
  std::vector<int> sceneSplits;
  sceneSplits.push_back(0);

  for(int k=1;k<K+1;++k){
    //std::cout << I.select(0,sceneSplits[k-1]).select(0,K-k).select(0,P_tot).item<int>() << "\n";
    sceneSplits.push_back(I.select(0,sceneSplits[k-1]).select(0,K-k).select(0,P_tot).item<int>());
    P_tot += (sceneSplits[k-1]-sceneSplits[k])*(sceneSplits[k-1]-sceneSplits[k]);

  }

  std::cout << P_tot << " ?= " << P.select(0,0).select(0,K-1).select(0,0) << "\n";
  std::cout << sceneSplits << "\n";

  std::ofstream resFile(outPath);
  int vsize = sceneSplits.size();
  for (int n=0; n<vsize; n++)
  {
      resFile << sceneSplits[n] << "\n";
  }

}
