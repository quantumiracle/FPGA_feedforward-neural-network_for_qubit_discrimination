///////////////////////////////////////
// Feedforward implementation of qubit2lay.py with C++
//using weights & bias data trained from qubit2lay.py Neural Network
//////////////////////////////////////
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
//#include "Timer.h"
#include<time.h>

using namespace std;

float relu(float x){
    if (x<0.0){
        x=0.0;
    }
    return x;
}
float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}
/*
void softmax(float &input, float &output){
    float sum=0;
    for(int n=0;n<input.size();n++){
        output.push_back(exp(input[n]));
        sum+=output[n];
    }
    for(int n=0;n<input.size();n++){
        output[n]/=m;
    }

}*/

//static Timer timer[1]={"time"};

int main(){
//set parameters
    int dat_dim=100;
    int lay1_dim=240;
    int lab_dim=2;
    int num_samples=10000;
    
    //timer[0].start();
    clock_t startTime,endTime;
	startTime = clock();

//load data
    int data[num_samples][dat_dim];

    ifstream file("bright.txt");  
    //ifstream file("dark.txt");
    char str;
    for(int i=0;i<num_samples;i++)
    {
        for(int j=0;j<dat_dim;j++)
        {
            data[i][j]=0;
        }
    }
    for(int i=0;i<num_samples;i++)
    {
        for(int j=0;j<dat_dim;j++)
        {
            //fscanf(file,"%d",&temp);
            if( ! file.eof() )
            file>>str;
            data[i][j]=int(str)-48;   //asc 0->48
        }
    }
    /*
    for(int i=0;i<num_samples;i++)
    {
        for(int j=0;j<dat_dim;j++)
        {
            cout<<data[i][j]<<" ";
        }
        cout<<endl;
    }
    */
    file.close();


//load bias
    double bias1[lay1_dim],bias2[lab_dim];

    //ifstream file("bias.txt");
    FILE *bias;  //float data in txt, ifstream: hard to tranvert data type from str to float(atof)
    bias = fopen("bias.txt","r");
    double temp;
    for (int i=0;i<lay1_dim;i++)
    if(fscanf(bias,"%lf",&temp)!=EOF)
    bias1[i]=temp;
    //cout<<x;

    //cout<<'\n';

    for (int i=0;i<lab_dim;i++)
    if(fscanf(bias,"%lf",&temp)!=EOF)
    bias2[i]=temp;
    //cout<<x;
    fclose(bias);

//load weights    
    double mat_dat_lay1[dat_dim][lay1_dim];
    double mat_lay1_lab[lay1_dim][lab_dim];

    FILE *weights;  //float data in txt, ifstream: hard to tranvert data type from str to float(atof)
    weights = fopen("weights.txt","r");
    for (int i=0;i<dat_dim;i++)
        for (int j=0;j<lay1_dim;j++){
            if(fscanf(weights,"%lf",&temp)!=EOF)
            mat_dat_lay1[i][j]=temp;
        }

    for (int i=0;i<lay1_dim;i++)
        for (int j=0;j<lab_dim;j++){
            if(fscanf(weights,"%lf",&temp)!=EOF)
            mat_lay1_lab[i][j]=temp;
        }
    fclose(weights);


//run matrix multiplication
    
    double lay1[lay1_dim];
    double lable[lab_dim];
    double lable_[lab_dim];
    int err=0;

    for (int k=0;k<num_samples;k++){
        for(int i=0;i<lay1_dim;i++){
            lay1[i]=0;
            for(int j=0;j<dat_dim;j++){
                lay1[i]+=mat_dat_lay1[j][i]*data[k][j];  //get lay1 value
                
            }
            lay1[i]=relu(lay1[i]+bias1[i]);   //nonlinear activation function

        }
        for(int i=0;i<lab_dim;i++){
            lable[i]=0;
            for(int j=0;j<lay1_dim;j++){
                lable[i]+=mat_lay1_lab[j][i]*lay1[j];

            }
            lable[i]+=bias2[i];
            lable_[i]=relu(lable[i]);

        }
        if (lable_[0]>=lable_[1]){
            err+=1;
        }
        /*
        for(int i=0;i<lab_dim;i++)
            cout<<lable_[i]<<' ';*/
    }
    cout<<"Errors:"<<' '<<err<<endl;
    cout<<"Total:"<<' '<<num_samples<<endl;

    endTime = clock();
	cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    //timer[0].stop();
    //cout<<timer[0].get_time();
}





            
