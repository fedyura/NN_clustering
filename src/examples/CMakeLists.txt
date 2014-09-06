ADD_EXECUTABLE (KohonenIris kohonen_iris.cpp)
TARGET_LINK_LIBRARIES(KohonenIris kohonen_nn weight_vector)

ADD_EXECUTABLE (NeuralGasIris neural_gas_iris.cpp)
TARGET_LINK_LIBRARIES(NeuralGasIris neural_gas weight_vector)
