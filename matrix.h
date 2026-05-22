#pragma once
#include "common.h"
#include <cmath>
#include <algorithm>

class Matrix{
public: 
    int _rows, _cols;
    // Using a flattened vector instead of 2D vector(mat[row][col] = row*numOfCols + col)
    vector<float> _mat;

    Matrix(int r, int c, int iv=0): _rows(r), _cols(c){
        _mat.resize(_rows*_cols, iv);
    }

    //access element at mat[r][c]
    float at(int row,int col) const{
        if(row >= _rows || col >= _cols){
            cout<<"Out of bound access\n";
            return -1;
        }
        return _mat[row*_cols + col];
    }

    //Set a particular element to some value
    void set(int row, int col, float value){
        if(row >= _rows || col >= _cols){
            cout<<"Out of bound access. Failed to set value\n";
        }else{
            _mat[row*_cols + col] = value;
        }
    }

    //Initialize the matrix with random values
    void randomize(float min=0.0,float max=1.0) {
        for(int i=0; i<_rows;i++){
            for(int j=0; j<_cols;j++){
                float val = ((float)rand()/RAND_MAX) * (max-min) + min;
                _mat[i*_cols+j] =  val;
            }
        }
    }

    //Transpose of matrix
    Matrix transpose() const{
        Matrix mtx(_cols,_rows);

        for(int r=0;r<_rows;r++){
            for(int c=0;c<_cols;c++){
                mtx.set(c,r,at(r,c));
            }
        }
        return mtx;
    }

    //Sum of matrix elements of same dimensions
    Matrix& operator+=(const Matrix& other) {
        assert(_cols==other._cols && _rows==other._rows);
        for(int i=0; i<_mat.size();i++){
            _mat[i] += other._mat[i];
        }
        return *this;
    }

    //Multiply each element with a value
    Matrix& operator*=(float val){
        for(int i=0;i<_mat.size();i++){
            _mat[i] = _mat[i]*val;
        }
        return *this;
    }

    //Access matrix elements as array
    float operator[](int val){
        return _mat[val];
    }

    //Modifies the LHS matrix in place( A=-B)
    Matrix& operator-=(const Matrix& m1){
        assert(this->_cols == m1._cols && this->_rows==m1._rows);
        for(int i=0; i<(_cols*_rows);i++){
            _mat[i] -= m1._mat[i];
        }
        return *this;
    }

    //Creates a copy, modifies copy and returns it keeping other matrices unaffected
    friend Matrix operator-(Matrix lhs ,const Matrix& rhs){
        lhs-=rhs;      //this modifies local copy lhs
        return lhs;
    }

    //Matrix multiplication
    static Matrix multiply(Matrix& m1, Matrix& m2){
        assert(m1._cols==m2._rows&& "Dimension mismatch for multiplication!");
        Matrix res(m1._rows, m2._cols);

        for(int i=0; i<m1._rows ;i++){
            for(int j=0; j<m2._cols;j++){
                res._mat[i*m2._cols + j] = 0;
                for(int k=0; k<m1._cols;k++){
                    res._mat[i*m2._cols + j] += m1._mat[i*m1._cols+k] * m2._mat[k*m2._cols+j];
                }
            }
        }
        return res;
    }

    //num of rows
    int rows(){
        return _rows;
    }

    //num of columns
    int columns(){
        return _cols;
    }

    void dimensions(){
        cout<<_rows<<"x"<<_cols<<endl;
    }

    //Print the matrix
    void print() const{
        for(int i=0;i<_rows;i++){
            for(int j=0;j<_cols;j++){
                cout<<_mat[i*_cols+j]<<" ";
            }
            cout<<"\n";
        }
    }

    //Sigmoid activation function
    float sigmoid(float x){
        return 1.f/(1.f + expf(-x));
    }

    void sigmoidActivate(){
        for(int i=0; i< (_rows*_cols); i++){
            _mat[i] = sigmoid(_mat[i]);
        }
    }

    float sigmoidDerivative(float sigmoid_output) {
        return sigmoid_output * (1.0f - sigmoid_output);
    }

    Matrix sigmoidDifferentiateLayer(){
        for(int i=0; i< (_rows*_cols); i++){
            _mat[i] = sigmoidDerivative(_mat[i]);
        }
        return *this;
    }

    //ReLU activation function
    float relu(float x) {
        return (x > 0) ? x : 0.0;
    }

    //Function to activate all the neurons in a layer(matrix) with relu
    void reluActivate(){
        for(int i=0; i< (_rows*_cols); i++){
            _mat[i] = relu(_mat[i]);
        }
    }

    //needed in backprop
    float reluDerivative(float x) {
        return (x > 0) ? 1.0 : 0.0;
    }

    //Function to differentiate all the neurons in a layer(matrix) with relu derivative
    void reluDifferentiateLayer(){
        for(int i=0; i< (_rows*_cols); i++){
            _mat[i] = reluDerivative(_mat[i]);
        }
    }

    /* 1. The Softmax activation function transforms a vector of raw scores (logits) 
          from the last layer of a neural network into a probability distribution.
       2. It works by calculating the exponential of each input value and dividing it 
          by the sum of all those exponentials. This ensures that:
            a. Every output value is between 0 and 1.
            b. The sum of all output values is exactly 1.0 (100%)
       3. Softmax is the standard for multi-class classification problems.
    */
    void softmax_activate() {
        // 1. Find the maximum value for numerical stability
        float maxVal = *max_element(_mat.begin(), _mat.end());
    
        // 2. Compute the exponentials and their sum
        float sum = 0.0;
        for (float& val : _mat) {
            val = exp(val - maxVal);
            sum += val;
        }

        // 3. Normalize the values so they sum to 1
        for (float& val : _mat) {
            val /= sum;
        }
    }
};