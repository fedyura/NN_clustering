#include <boost/format.hpp>
#include <ctime>
#include <iostream>
#include <limits>
#include <logger/logger.hpp>
#include <neural_network/KohonenNN.hpp>
#include <memory>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace nn
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("KohonenNetwork");

    KohonenNN::KohonenNN(uint32_t num_clusters, uint32_t num_dimensions, alr::KohonenParameters kp, double min_potential, NetworkInitType nnit, neuron::NeuronType nt)
        : m_NumClusters(num_clusters)
        , m_NumDimensions(num_dimensions)
        , m_Kp(kp)
        , m_MinPotential(min_potential)
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
    KohonenNN::KohonenNN(bool is_initialized, uint32_t num_clusters, uint32_t num_dimensions, alr::KohonenParameters kp, double min_potential, NetworkInitType nnit, neuron::NeuronType nt)
        : m_NumClusters(num_clusters)
        , m_NumDimensions(num_dimensions)
        , m_Kp(kp)
        , m_MinPotential(min_potential)
        , m_InitNetType(nnit)
        , m_NeuronType(nt)
        , m_NumNeuronWinner(0)
        , m_IterNumber(0)
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
            for (uint32_t j = 0; j < m_NumDimensions; j++)
            {    
                coords[j] = ((double) rand() / (RAND_MAX));
                log_netw->debug((boost::format("%g,") % coords[j]).str(), true);
            }
            log_netw->debug("\n", true);
            
            switch(m_NeuronType)
            {
                case neuron::NeuronType::EUCLIDEAN:
                {
                    wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);  
                    neurons[i] = neuron::KohonenNeuron(sWeightVector, m_MinPotential); //create neurons 
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
        double min_dist = std::numeric_limits<double>::max(), cur_dist = 0;
        for (uint32_t i = 0; i < m_NumClusters; i++)
        {
            neuron::KohonenNeuron& knn = m_Neurons.at(i);
            if (!knn.mayBeWinner())
                continue;           
            cur_dist = knn.setCurPointDist(p);
            if (cur_dist < min_dist)
            {
                min_dist = cur_dist;
                m_NumNeuronWinner = i;
            }
        }        
    }

    void KohonenNN::updateWeights(const wv::Point* p, const alr::AbstractAdaptLearnRate* alr)
    {
        //update neuron winner
        neuron::KohonenNeuron& kn = m_Neurons.at(m_NumNeuronWinner);
        kn.getWv()->updateWeightVector(p, alr, kn.setCurPointDist(m_Neurons.at(m_NumNeuronWinner).getWv()));
        kn.decreasePotential();
        
        //update other neurons
        for (uint32_t i = 0; i < m_NumClusters; i++)
        {
            if (i == m_NumNeuronWinner)
                continue;
            
            neuron::KohonenNeuron& kn = m_Neurons.at(i);
            kn.getWv()->updateWeightVector(p, alr, kn.setCurPointDist(m_Neurons.at(m_NumNeuronWinner).getWv()));
            kn.increasePotential(m_NumClusters);
        }
    }

    //return true if we need to continue training, false - otherwise
    bool KohonenNN::trainOneEpoch(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon)
    {
        //Erase offset vector
        for (uint32_t i = 0; i < m_NumClusters; i++)
            m_Neurons.at(i).getWv()->eraseOffset();

        alr::AdaptLearnRateKohonenSchema alrks(m_IterNumber, m_Kp);
        //make iteration
        for (const auto p: points)
        {
            findWinner(p.get());
            updateWeights(p.get(), &alrks);
        }
        
        //check offset value
        double summary_offset_value = 0;
        for (uint32_t i = 0; i < m_NumClusters; i++)
        {
            summary_offset_value += m_Neurons.at(i).getWv()->getOffsetValue();
        }

        log_netw->info(" --------------------------------------------------------------------");
        log_netw->info((boost::format("After %d iteration") % m_IterNumber).str());
        log_netw->info((boost::format("Summary offset value = %g") % summary_offset_value).str()); 
        for (uint32_t i = 0; i < m_NumClusters; i++)
        {
            for (uint32_t j = 0; j < m_NumDimensions; j++)
                log_netw->info((boost::format("%g,") % m_Neurons.at(i).getWv()->getConcreteCoord(j)).str(), true);
            log_netw->info("\n", true);
        }
        m_IterNumber++;
        return (summary_offset_value / m_NumClusters > epsilon);    
    }

    void KohonenNN::trainNetwork(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon)
    {
        while (trainOneEpoch(points, epsilon))
        {
        }
    }

    uint32_t KohonenNN::getCluster(const wv::Point* p)
    {
        findWinner(p);
        return m_NumNeuronWinner;
    }
}
