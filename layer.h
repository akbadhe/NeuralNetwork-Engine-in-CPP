#pragma once
#include "common.h"
#include "matrix.h"

class Layer{
public:
    Matrix weights;
    Matrix bias;
    Matrix layer_input;
    Matrix layer_output;
    Matrix input_gradient;
    float LR;
    bool isOutputLayer;

    //Contructor to initilialize all matrices with rows and cols and random weights
    Layer(int in_dimensions, int out_dimensions, bool layerType=false)
       :weights(in_dimensions,out_dimensions),
        bias(1,out_dimensions),
        layer_input(1,in_dimensions),
        layer_output(1,out_dimensions),
        input_gradient(1,in_dimensions),
        isOutputLayer(layerType){

        weights.randomize(-0.5f,0.5f);
        LR=1;
    };

    //Do the forward pass and predict
    Matrix forward(Matrix& input){
        layer_input=input;
        Matrix net = Matrix::multiply(layer_input,weights);
        net += bias;
        
        net.sigmoidActivate();    //apply activation function to break linearity
        layer_output=net;

        return layer_output;
    }


    Matrix backpropagate(Matrix& out_gradient) {
        // 1. Get the activation derivative matrix (Shape: 1 x outputs)
        Matrix activationDerivative = layer_output.sigmoidDifferentiateLayer();

        // 2. Calculate local gradient Delta (Shape: 1 x outputs)
        // Note: This MUST be element-wise multiplication, NOT Matrix::multiply
        Matrix delta(out_gradient._rows, out_gradient._cols);
        for (int i = 0; i < delta._rows * delta._cols; i++) {
            delta._mat[i] = out_gradient._mat[i] * activationDerivative._mat[i];
        }

        // 3. Calculate Weight Gradients (Shape: inputs x outputs)
        Matrix inpTranspose = layer_input.transpose(); // Shape: inputs x 1
        Matrix deltaWeight = Matrix::multiply(inpTranspose, delta); // (inputs x 1) * (1 x outputs) = (inputs x outputs)

        // 4. Calculate Bias Gradients (Shape: 1 x outputs)
        // The gradient for the bias is exactly delta itself
        Matrix deltaBias = delta; 

        // 5. Apply Learning Rate and Update Parameters
        deltaWeight *= LR;
        weights -= deltaWeight; 

        deltaBias *= LR;
        bias -= deltaBias;

        // 6. Calculate Input Gradient to pass back to the previous layer (Shape: 1 x inputs)
        Matrix wtranspose = weights.transpose(); // Shape: outputs x inputs
        Matrix inp_gradient = Matrix::multiply(delta, wtranspose); // (1 x outputs) * (outputs x inputs) = (1 x inputs)
    
        input_gradient=inp_gradient;
        return inp_gradient;
    }

};