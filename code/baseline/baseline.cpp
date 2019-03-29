#include <torch/torch.h>
#include <iostream>
#include <string>
#include <stdlib.h>     /* strtod */



int main(int argc, char *argv[]){



  std::string matrixPath = argv[1];
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int pMax = N*N;
  int cuda = atoi(argv[4]);

  torch::Tensor simMat = torch::zeros({N,N});

  std::ifstream f(matrixPath);
  float elem;
  for (int i=0;i<N;++i){
    for (int j=0;j<N;++j){
        f >> elem;
        simMat[i][j] = elem;
    }
  }

  torch::Tensor C = torch::ones({N,K,pMax})*std::numeric_limits<double>::quiet_NaN();
  torch::Tensor I = torch::ones({N,K,pMax})*std::numeric_limits<double>::quiet_NaN();
  torch::Tensor P = torch::ones({N,K,pMax})*std::numeric_limits<double>::quiet_NaN();
  torch::Tensor G = torch::ones({N-1})*std::numeric_limits<double>::quiet_NaN();

  if(cuda){

    C = C.cuda();
    I = I .cuda();
    P = P.cuda();
    G = G.cuda();
    simMat = simMat.cuda();
  }

  I.slice(0,0,N).select(1,0).slice(1,0,pMax) = N;

  for(int k=0;k<K;++k){
    std::cout << "k = " << k << "\n";
    for(int n=1;n<N+1;++n){
      std::cout << "\tn = " << n << "\n";

      P.select(0,n-1).select(0,0) = (N-n+1)*(N-n+1);
      C.select(0,n-1).select(0,0).slice(0,n-1,n*n) = simMat.slice(0,n,N).slice(1,n,N).sum()/(torch::arange(n-1,n*n)+(N-n+1)*(N-n+1)-N);

      for(int p=n-1;p<n*n;++p){

        if (k>0){

          std::cout << "k>0" << "\n";

        }


      }



    }

  }



}
