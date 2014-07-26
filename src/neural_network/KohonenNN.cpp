#include <ctime>
#include <neural_network/KohonenNN.hpp>
#include <memory>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace nn
{
    KohonenNN::KohonenNN(uint32_t num_clusters, uint32_t num_dimensions, alr::KohonenParameters kp, NetworkInitType nnit, neuron::NeuronType nt)
        : m_NumClusters(num_clusters)
        , m_NumDimensions(num_dimensions)
        , m_Kp(kp)
        , m_InitNetType(nnit)
        , m_NeuronType(nt)
        , m_NumNeuronWinner(0)
        , m_IterNumber(0)
    {
        if (m_NumClusters < 2)
            throw std::runtime_error("Can't construct neural network. The number of clusters must be more than 1.");
        if (m_NumDimensions < 1)
            throw std::runtime_error("Can't construct neural network. The number of dimensions must be more than 0.");
        initializeNN();    
    }
    
    //constructor for class descendant
    KohonenNN::KohonenNN(bool is_initialized, uint32_t num_clusters, uint32_t num_dimensions, alr::KohonenParameters kp, NetworkInitType nnit, neuron::NeuronType nt)
        : m_NumClusters(num_clusters)
        , m_NumDimensions(num_dimensions)
        , m_Kp(kp)
        , m_InitNetType(nnit)
        , m_NeuronType(nt)
        , m_NumNeuronWinner(0)
    {
        if (m_NumClusters < 2)
            throw std::runtime_error("Can't construct neural network. The number of clusters must be more than 1.");
        if (m_NumDimensions < 1)
            throw std::runtime_error("Can't construct neural network. The number of dimensions must be more than 0.");
    }

    void KohonenNN::initializeNN()
    {
        srand(std::time(NULL));
        cont::StaticArray<double> coords(m_NumDimensions);
        cont::StaticArray<neuron::KohonenNeuron> neurons(m_NumClusters);
        
        for (uint32_t i = 0; i < neurons.size(); i++)
        {
            for (uint32_t i = 0; i < m_NumDimensions; i++)
                coords[i] = ((double) rand() / (RAND_MAX));
            
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
                    throw std::runtime_error("Can't construct neural network. Incorrect neuron type.");
                    break;
                }
            }
        }
        m_Neurons = neurons;
    }

    KohonenNN::~KohonenNN()
    {
        for (uint32_t i = 0; i < m_NumClusters; i++)
            if (m_Neurons[i].getWv() != NULL)
            {
                delete m_Neurons[i].getWv();
                m_Neurons[i].setZeroPointer();
            }
    }

    void KohonenNN::findWinner(const wv::Point* p)
    {
        double min_dist = m_Neurons.at(0).setCurPointDist(p), cur_dist = 0;
        for (uint32_t i = 1; i < m_NumClusters; i++)
        {
            cur_dist = m_Neurons.at(i).setCurPointDist(p);
            if (cur_dist < min_dist)
            {
                min_dist = cur_dist;
                m_NumNeuronWinner = i;
            }
        }        
    }

    void KohonenNN::updateWeights()
    {
        
        
    }
}
