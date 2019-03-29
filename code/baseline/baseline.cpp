#include <torch/torch.h>
#include <iostream>
#include <string>

int main(int argc, char *argv[]){



  std::string matrixPath = argv[1];
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
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


  

}
