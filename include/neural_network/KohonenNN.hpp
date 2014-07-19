#ifndef __NEURAL_NETWORK_KOHONEN_NN_HPP__
#define __NEURAL_NETWORK_KOHONEN_NN_HPP__

#include <container/StaticArray.hpp>
#include <neuron/KohonenNeuron.hpp>

namespace nn
{
    enum NetworkInitType
    {
        RANDOM = 1        
    };
    
    class KohonenNN
    {
    public:
        void initializeNN(); //init nerons by default weight vectors
        void findWinner(const wv::Point* p);
        void updateWeights();
        void train();
        void getMapError();
        
        //get methods
        uint32_t numClusters()
        {
            return m_NumClusters;
        }

        uint32_t numDimensions()
        {
            return m_NumDimensions;
        }

        NetworkInitType initNetType()
        {
            return m_InitNetType;
        }

        neuron::NeuronType neuronType()
        {
            return m_NeuronType;
        }

        neuron::KohonenNeuron getNeuron(uint32_t number)
        {
            return m_Neurons.at(number);
        }

        virtual ~KohonenNN();

        KohonenNN(uint32_t num_clusters, uint32_t num_dimensions, NetworkInitType nnit = NetworkInitType::RANDOM, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN);

    private:
        uint32_t m_NumClusters;        //Number of clusters equal number of neurons
        uint32_t m_NumDimensions;
        NetworkInitType m_InitNetType; //Type of network initialization
        neuron::NeuronType m_NeuronType;       //Type of the neuron
        cont::StaticArray<neuron::KohonenNeuron> m_Neurons;
        uint32_t m_NumNeuronWinner;
    };
}

#endif //__NEURAL_NETWORK_KOHONEN_NN_HPP__
