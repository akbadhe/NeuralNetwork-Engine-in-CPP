#include <initializer_list>  //For variable num of args in ctor
#include "common.h"
#include "matrix.h"
#include "layer.h"

class Network{
public:
    
    vector<int> topology;       //Network topology(num of neurons in each layer)
    vector<Layer> layers;       //All layers stored here

    Network(initializer_list<int> list){
        if(list.size() < 3){
            cout<<"Invalid topology. (Atleast 3 layers required)"<<endl;
            exit(EXIT_FAILURE);
        }
        topology = list;
        createNetwork();
    }

    //creates network with given topology
    void createNetwork(){
        int sptr=0, eptr=1;

        while(eptr < (topology.size())){
            if(eptr == (topology.size()-1)){               //Output Layer
                Layer l(topology[sptr++],topology[eptr++],true);
                layers.push_back(l);
            }else{                                         //Hidden Layers
                Layer l(topology[sptr++],topology[eptr++]);
                layers.push_back(l);
            }
        }
    }

    //Train the neural network
    void train(vector<Matrix> inputs, vector<Matrix> labels, int EPOCHS){
        for(int epoch=0;epoch< EPOCHS ;epoch++){
            if(epoch%500==0) cout<<"\nEpoch:"<<epoch<<endl;

            for(int i=0;i<inputs.size();i++){
                // A single forward training step look like this:
                Matrix final_predictions(1, topology[topology.size()-1]);

                for(int j=0; j< layers.size();j++){
                    if(j==0){                            //for 1st hidden layer
                        Matrix activation = layers[j].forward(inputs[i]);
                    }else{                               //for rest of the layers
                        Matrix activation = layers[j].forward(layers[j-1].layer_output);
                        if(layers[j].isOutputLayer){     //if output layer then save the prediction for loss calculation
                            final_predictions = activation;
                        }
                    }
                }
            
                // Calculating loss derivative (Output Gradient)
                Matrix loss_gradient = final_predictions - labels[i];
                if(epoch%500==0){
                    cout<<"Output for input:";
                    inputs[i].print();
                    cout<<"is :";
                    final_predictions.print();
                }  

                // A single backward step to update all network variables:             
                for(int j=layers.size()-1 ; j>=0 ; j--){
                    if(j==(layers.size()-1)){             //for output layer
                        Matrix gradient = layers[j].backpropagate(loss_gradient);
                    }else{                               //for rest of the layers
                        Matrix gradient = layers[j].backpropagate(layers[j+1].input_gradient);
                    }
                }
            }
        }
    }

    //Predict the output for a given input
    Matrix predict(Matrix input){
        Matrix final_predictions(1, topology[topology.size()-1]);
        //for(auto layer:layers){
        for(int j=0; j< layers.size();j++){
            if(j==0){                            //for 1st hidden layer
                Matrix activation = layers[j].forward(input);
            }else{                               //for rest of the layers
                Matrix activation = layers[j].forward(layers[j-1].layer_output);
                if(layers[j].isOutputLayer){     //if output layer then save the prediction
                    final_predictions = activation;
                }
            }
        }
        cout<<"\nPredicted value: "<<endl;
        final_predictions.print();
        return final_predictions;
    }
};

