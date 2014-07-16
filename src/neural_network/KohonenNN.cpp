#include <ctime>
#include <neural_network/KohonenNN.hpp>
#include <memory>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace nn
{
    KohonenNN::KohonenNN(uint32_t num_clusters, uint32_t num_dimensions, NetworkInitType nnit, neuron::NeuronType nt)
        :m_NumClusters(num_clusters)
        ,m_NumDimensions(num_dimensions)
        ,m_InitNetType(nnit)
        ,m_NeuronType(nt)
    {
        srand(std::time(NULL));
        initializeNN();
    }

    void KohonenNN::initializeNN()
    {
        cont::StaticArray<double> coords(m_NumDimensions);
        cont::StaticArray<neuron::KohonenNeuron> neurons(m_NumClusters);
        
        for (uint32_t i = 0; i < neurons.size(); i++)
        {
            switch(m_NeuronType)
            {
                case neuron::NeuronType::EUCLIDEAN:
                {
                    wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);  
                    neurons[i] = neuron::KohonenNeuron(sWeightVector); //create neurons 
                    break;
                }
                default:
                {
                    throw std::runtime_error("Incorrect neuron type");
                    break;
                }
            }
            neurons[i].initRandomValues();                           //initialize neurons
        }
        m_Neurons = neurons;
    }

    KohonenNN::~KohonenNN()
    {
        for (uint32_t i = 0; i < m_NumClusters; i++)
            delete m_Neurons[i].getWv();
    }
}
