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
#include"fcl.h"
#include"sdx.h"

#include<time.h>

using namespace std;




int main(){
//set parameters
    const int dat_dim=100;
    const int lay1_dim=240;
    const int lab_dim=2;
    const int num_samples=10000;

    //timer[0].start();
    clock_t startTime,endTime;
	//startTime = clock();


	int* input_layer	= (int*) MEM_ALLOC(100*sizeof(int));
	float* output_layer_1	= (float*) MEM_ALLOC(240*sizeof(float));
    float* output_layer_2	= (float*) MEM_ALLOC(2*sizeof(float));




    //load bias

    float* bias1	= (float*) MEM_ALLOC(240*sizeof(float));
	float* bias2	= (float*) MEM_ALLOC(2*sizeof(float));


    FILE *bias;
    bias = fopen("bias.txt","r");
    float temp;
    for (int i=0;i<lay1_dim;i++)
    if(fscanf(bias,"%f",&temp)!=EOF)
    bias1[i]=float(temp);


    for (int i=0;i<lab_dim;i++)
    if(fscanf(bias,"%f",&temp)!=EOF)
    bias2[i]=float(temp);
    //cout<<x;
    fclose(bias);

//load weights

    float* w_1	= (float*) MEM_ALLOC(100*240*sizeof(float));
	float* w_2	= (float*) MEM_ALLOC(240*2*sizeof(float));


	//float w_1[dat_dim*lay1_dim];
   // float w_2[lay1_dim*lab_dim];

    FILE *weights;  //float data in txt, ifstream: hard to tranvert data type from str to float(atof)
    weights = fopen("weights.txt","r");
    for (int i=0;i<dat_dim;i++){
        for (int j=0;j<lay1_dim;j++){
            if(fscanf(weights,"%f",&temp)!=EOF)
           w_1[i*lay1_dim+j]=float(temp);
        }
    }

    for (int i=0;i<lay1_dim;i++){
        for (int j=0;j<lab_dim;j++){
            if(fscanf(weights,"%f",&temp)!=EOF)
            w_2[i*lab_dim+j]=float(temp);
        }
    }
    fclose(weights);




//run matrix multiplication

   /* double lay1[lay1_dim];
    double lable[lab_dim];
    double lable_[lab_dim];
    int err=0;*/
	int err=0;

	ifstream file("bright.txt");
    char str;
    //int init=1;
	startTime = clock();
	for (int k=0;k<num_samples;k++){


    	    //ifstream file("dark.txt");
    	    for(int j=0;j<dat_dim;j++)
    	          {
    	              //fscanf(file,"%d",&temp);
    	              if( ! file.eof() )
    	              file>>str;
    	              input_layer[j]=(int(str)-48);   //asc 0->48
    	          }


    	fcl_1(input_layer,w_1,bias1,output_layer_1);
        fcl_2(output_layer_1,w_2,bias2,output_layer_2);
        if (output_layer_2[0]>=output_layer_2[1]){
            err+=1;
        }
        /*
        for(int i=0;i<lab_dim;i++)
            cout<<lable_[i]<<' ';*/
    }

    file.close();
/*

    MEM_FREE(input_layer);
    MEM_FREE(output_layer_1);
    MEM_FREE(output_layer_2);
    MEM_FREE(w_1);
    MEM_FREE(w_2);
    MEM_FREE(bias1);
    MEM_FREE(bias2);
*/
    cout<<"Errors:"<<' '<<err<<endl;
    cout<<"Total:"<<' '<<num_samples<<endl;

    endTime = clock();
	cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    //timer[0].stop();
    //cout<<timer[0].get_time();
}





