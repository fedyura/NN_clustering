#ifndef __NEURAL_NETWORK_KOHONEN_NN_HPP__
#define __NEURAL_NETWORK_KOHONEN_NN_HPP__

#include <adapt_learn_rate/AdaptLearnRateKohonenSchema.hpp>
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

        uint32_t numNeuronWinner()
        {
            return m_NumNeuronWinner;
        }

        virtual ~KohonenNN();

        KohonenNN(uint32_t num_clusters, uint32_t num_dimensions, alr::KohonenParameters kp, NetworkInitType nnit = NetworkInitType::RANDOM, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN);

    protected:  
        //only for class-descendant
        KohonenNN(bool is_initialized, uint32_t num_clusters, uint32_t num_dimensions, alr::KohonenParameters kp, NetworkInitType nnit = NetworkInitType::RANDOM, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN);
        
        cont::StaticArray<neuron::KohonenNeuron> m_Neurons;
        
        //methods described network algorithm. It might be overrided in derived class  
        void findWinner(const wv::Point* p);
        void updateWeights();
        void train();
        void getMapError();
    
    private:
        uint32_t m_NumClusters;        //Number of clusters equal number of neurons
        uint32_t m_NumDimensions;
        alr::KohonenParameters m_Kp;
        NetworkInitType m_InitNetType; //Type of network initialization
        neuron::NeuronType m_NeuronType;       //Type of the neuron
        uint32_t m_NumNeuronWinner;
        uint32_t m_IterNumber;         //Number of training iteration of neural network

        void initializeNN(); //init neurons by default weight vectors
    };
}

#endif //__NEURAL_NETWORK_KOHONEN_NN_HPP__
