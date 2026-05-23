#include "common.h"
#include "layer.h"
#include "matrix.h"
#include "network.h"

int main(){

    Matrix inp1(1,2);
    inp1.set(0,0,0);
    inp1.set(0,1,0);

    Matrix inp2(1,2);
    inp2.set(0,0,0);
    inp2.set(0,1,1);

    Matrix inp3(1,2);
    inp3.set(0,0,1);
    inp3.set(0,1,0);

    Matrix inp4(1,2);
    inp4.set(0,0,1);
    inp4.set(0,1,1);

    Matrix out1(1,1);
    out1.set(0,0,0);

    Matrix out2(1,1);
    out2.set(0,0,1);

    Matrix out3(1,1);
    out3.set(0,0,1);

    Matrix out4(1,1);
    out4.set(0,0,0);

    vector<Matrix> inputs={inp1,inp2,inp3,inp4};
    vector<Matrix> labels = {out1,out2,out3,out4};

    Network net = {2,4,1};
    net.train(inputs,labels,2000);
    net.predict(inp2);

    return 0;
}