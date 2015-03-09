ADD_EXECUTABLE (KohonenIris kohonen_iris.cpp)
TARGET_LINK_LIBRARIES(KohonenIris kohonen_nn weight_vector)

ADD_EXECUTABLE (KohonenSynthetic kohonen_2D_synthetic.cpp)
TARGET_LINK_LIBRARIES(KohonenSynthetic kohonen_nn weight_vector)

ADD_EXECUTABLE (NeuralGasIris neural_gas_iris.cpp)
TARGET_LINK_LIBRARIES(NeuralGasIris neural_gas weight_vector)

ADD_EXECUTABLE (SoinnIris soinn_iris.cpp)
TARGET_LINK_LIBRARIES(SoinnIris soinn weight_vector)

ADD_EXECUTABLE (SoinnSynthetic soinn_2D_synthetic.cpp)
TARGET_LINK_LIBRARIES(SoinnSynthetic soinn weight_vector)

ADD_EXECUTABLE (ESoinnIris esoinn_iris.cpp)
TARGET_LINK_LIBRARIES(ESoinnIris esoinn weight_vector)

ADD_EXECUTABLE (ESoinnForest esoinn_forest.cpp)
TARGET_LINK_LIBRARIES(ESoinnForest esoinn weight_vector)

ADD_EXECUTABLE (ESoinnSynthetic esoinn_2D_synthetic.cpp)
TARGET_LINK_LIBRARIES(ESoinnSynthetic esoinn weight_vector)
