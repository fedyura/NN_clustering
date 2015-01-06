ADD_LIBRARY(kohonen_nn STATIC KohonenNN.cpp)
ADD_LIBRARY(neural_gas STATIC NeuralGas.cpp)
ADD_LIBRARY(soinn      STATIC Soinn.cpp    )

ADD_SUBDIRECTORY(test)
