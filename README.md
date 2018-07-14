# FPGA_feedforward-neural-network_for_qubit_discrimination
```main.cpp```: main function to execute qubit discrimination test with C++ feedforward neural network. \
to use: ```g++ -o main main.cpp```  & ```./main```

```qubit2lay.py```: Python qubit discirmination function with 2 layers neural network, generating weights.txt and bias.txt. The 
structure of neural network in Python and C++ is exactly the same.

```trans.py```: transform the origin data from .gzip to txt, so as to be read by main.cpp.

```fcl.cpp/fcl.h```: fully connected layers computation functions implemented on FPGA instead of ARM.

```bright.txt/bright1.txt```: testing data, only contains bright states data.
