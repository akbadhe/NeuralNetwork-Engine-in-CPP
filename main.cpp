#include "common.h"
#include "layer.h"
#include "matrix.h"

#define EPOCHS 2000

void train(vector<Matrix> inputs, vector<Matrix> labels){

    // Instantiating a clean modular network architecture
    Layer hidden_layer(2, 3); // 2 inputs -> 3 hidden units
    Layer output_layer(3, 1); // 3 hidden units -> 1 output unit
    
    cout<<"Training the network"<<endl;
    for(int epoch=0;epoch< EPOCHS ;epoch++){
        if(epoch%500==0) cout<<"\nEpoch:"<<epoch<<endl;

        for(int i=0;i<inputs.size();i++){
            // A single forward training step look like this:
            Matrix hidden_activations = hidden_layer.forward(inputs[i]);
            Matrix final_predictions  = output_layer.forward(hidden_activations);
            
            // Calculating loss derivative (Output Gradient)
            Matrix loss_gradient = final_predictions - labels[i];
            if(epoch%500==0){
                cout<<"Output for input:";
                inputs[i].print();
                cout<<"is :";
                final_predictions.print();
            }  

            // A single backward step to update all network variables:
            Matrix hidden_gradient = output_layer.backpropagate(loss_gradient);
            hidden_layer.backpropagate(hidden_gradient);
        }
    }
}

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

    train(inputs,labels);

    return 0;
}