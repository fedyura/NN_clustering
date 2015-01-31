ADD_EXECUTABLE (KohonenIris kohonen_iris.cpp)
TARGET_LINK_LIBRARIES(KohonenIris kohonen_nn weight_vector)

ADD_EXECUTABLE (NeuralGasIris neural_gas_iris.cpp)
TARGET_LINK_LIBRARIES(NeuralGasIris neural_gas weight_vector)

ADD_EXECUTABLE (SoinnIris soinn_iris.cpp)
TARGET_LINK_LIBRARIES(SoinnIris soinn weight_vector)

ADD_EXECUTABLE (ESoinnIris esoinn_iris.cpp)
TARGET_LINK_LIBRARIES(ESoinnIris esoinn weight_vector)
