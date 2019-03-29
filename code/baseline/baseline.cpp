#include <torch/torch.h>
#include <iostream>
#include <string>
#include <stdlib.h>     /* strtod */



int main(int argc, char *argv[]){



  std::string matrixPath = argv[1];
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int pMax = N*N;
  std::string device = argv[4];

  torch::Tensor simMat = torch::zeros({N,N},device);

  std::ifstream f(matrixPath);
  float elem;
  for (int i=0;i<N;++i){
    for (int j=0;j<N;++j){
        f >> elem;
        simMat[i][j] = elem;
    }
  }

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

  torch::Tensor simMat_exp;
  torch::Tensor sumVec;

  torch::Tensor ind0;
  torch::Tensor ind1;
  torch::Tensor ind2;

  torch::Tensor mask;

  torch::Tensor summedAreas;

  torch::Tensor P_k_sel;
  torch::Tensor C_k_sel;

  torch::Tensor G_paral;

  I.slice(0,0,N).select(1,0).slice(1,0,pMax) = N;

  for(int k=0;k<K;++k){
    std::cout << "k = " << k << "\n";
    for(int n=1;n<N+1;++n){
      std::cout << "\tn = " << n << "\n";

      P.select(0,n-1).select(0,0) = (N-n+1)*(N-n+1);
      C.select(0,n-1).select(0,0).slice(0,n-1,n*n) = simMat.slice(0,n,N).slice(1,n,N).sum()/(torch::arange(n-1,n*n)+(N-n+1)*(N-n+1)-N);

      for(int p=n-1;p<n*n;++p){

        if (k>0){


          // Computation of the numerator
          std::cout << "Ind init" << "\n";
          ind0 = nValues.slice(0,n-1,N-1).unsqueeze(1).unsqueeze(2).expand({N-n,N-n,N-n});
          ind1 = nValues.slice(0,n-1,N-1).unsqueeze(0).unsqueeze(2).expand({N-n,N-n,N-n});
          ind2 = nValues.slice(0,n-1,N-1).unsqueeze(0).unsqueeze(1).expand({N-n,N-n,N-n});

          std::cout << "sim Mat expand and mask" << "\n";
          simMat_exp = simMat.slice(0,n-1,N-1).slice(1,n-1,N-1).unsqueeze(0).expand({N-n,N-n,N-n});
          simMat_exp = simMat_exp*((ind1 <= ind0)*(ind2 <= ind0)).to(simMat_exp.dtype());

          std::cout << "sim mat sum" << "\n";
          sumVec = simMat_exp.sum(2).sum(1);

          //Computation of the denominator
          std::cout << "Ind init" << "\n";
          ind0 = nValues.slice(0,n-1,N-1).unsqueeze(1).expand({N-n,pMax});
          ind1 = pValues.unsqueeze(0).expand({N-n,pMax});

          std::cout << "Computing the mask" << "\n";
          mask = (ind1 == p+(ind0-n+1)*(ind0-n+1));

          ind0 = nValues.slice(0,n-1,N-1);

          summedAreas = p+(ind0-n+1)*(ind0-n+1);

          std::cout << "Slicing and selecting P and C" << "\n";
          P_k_sel = P.slice(0,n-1,N-1).select(1,k-1).masked_select(mask);
          C_k_sel = C.slice(0,n-1,N-1).select(1,k-1).masked_select(mask);

          std::cout << "Computing G" << "\n";
          G_paral = sumVec/(summedAreas+P_k_sel-N)+C_k_sel;

          std::cout << "Computing the min" << "\n";
          int argmin,min;
          if (G_paral.size(0) == 0){
            min = 0;
            argmin=0+n-1;
          }else{
            min = G_paral.min().item<float>();
            argmin = G_paral.argmin().item<int>();
          }

          C.select(0,n-1).select(0,k).select(0,p) = min;
          I.select(0,n-1).select(0,k).select(0,p) = argmin;

          std::cout << "Computing the area" << "\n";
          int b = ((argmin-n+1)*(argmin-n+1));

          if(p+b < P.size(2)){
            P.select(0,n-1).select(0,k).select(0,p) = b+P.select(0,argmin).select(0,k-1).select(0,p+b);

          }
        }


      }



    }

  }



}
