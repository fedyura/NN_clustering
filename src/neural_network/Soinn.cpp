#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <algorithm>
#include <boost/format.hpp>
#include <fstream>
#include <logger/logger.hpp>
#include <neural_network/Soinn.hpp>

namespace nn
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("Soinn");
    
    Soinn::Soinn(uint32_t num_dimensions, double alpha1, double alpha2, double alpha3, double betta, double gamma, double age_max, uint32_t lambda, NetworkStopCriterion nnit, neuron::NeuronType nt)
    : m_NumDimensions(num_dimensions)
    , m_Alpha1(alpha1)
    , m_Alpha2(alpha2)
    , m_Alpha3(alpha3)
    , m_Betta(betta)
    , m_Gamma(gamma)
    , m_AgeMax(age_max)
    , m_Lambda(lambda)
    , m_NetStop(nnit)
    , m_NeuronType(nt)
    , m_NumWinner(0)
    , m_NumSecondWinner(0)
    {
        if (m_NumDimensions < 1)
            throw std::runtime_error("Can't construct neural network. The number of dimensions must be more than 0.");
    }

    Soinn::~Soinn()
    {
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
            if (m_Neurons[i].getWv() != NULL)
            {
                delete m_Neurons[i].getWv();
                m_Neurons[i].setZeroPointer();
            }
    }
    
    void Soinn::initialize(const std::pair<wv::Point*, wv::Point*>& points)
    {
        if (m_NumDimensions != points.first->getNumDimensions())
            throw std::runtime_error("Number of dimensions for data doesn't correspond dimension of neural network");
        cont::StaticArray<double> coords1(m_NumDimensions);
        cont::StaticArray<double> coords2(m_NumDimensions);

        for (uint32_t i = 0; i < m_NumDimensions; i++)
        {
            coords1[i] = points.first->getConcreteCoord(i);
            coords2[i] = points.second->getConcreteCoord(i);
        }
        
        wv::WeightVectorContEuclidean* sWeightVector1 = new wv::WeightVectorContEuclidean(coords1);
        wv::WeightVectorContEuclidean* sWeightVector2 = new wv::WeightVectorContEuclidean(coords2);
        m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector1));
        m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector2));
        m_Neurons[0].updateEdge(1);
        m_Neurons[1].updateEdge(0);
    }

    std::pair<double, double> Soinn::findWinners(const wv::Point* p)
    {
        double min_dist_main = std::numeric_limits<double>::max(), min_dist_sec = std::numeric_limits<double>::max();
        double cur_dist = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            cur_dist = m_Neurons[i].setCurPointDist(p);
            if (cur_dist < min_dist_main)
            {
                min_dist_sec = min_dist_main;
                min_dist_main = cur_dist;
                m_NumSecondWinner = m_NumWinner;
                m_NumWinner = i;
            }
            else if (cur_dist < min_dist_sec)
            {
                min_dist_sec = cur_dist;
                m_NumSecondWinner = i;
            }
        }
        return std::make_pair(min_dist_main, min_dist_sec);
    }

    double Soinn::EvalThreshold(uint32_t num_neuron)
    {
        std::vector<uint32_t> neighbours = m_Neurons.at(num_neuron).getNeighbours();
        double threshold = 0;

        //If neuron has any neighbours find maximum of distance between neighbours (max within cluster distance)
        if (neighbours.size() > 0)
        {
            double max_dist = 0, cur_dist = 0;
            for (uint32_t num: neighbours)
            {
                cur_dist = m_Neurons[num_neuron].setCurPointDist(m_Neurons.at(num).getWv());
                if (cur_dist > max_dist) 
                    max_dist = cur_dist;
            }
            threshold = max_dist;
        }
        //if neuron doesn't have any neighbours find mininum of distance between this and other neurons (min inner cluster distance)
        else
        {
            double min_dist = std::numeric_limits<double>::max(), cur_dist = 0;
            //здесь тоже надо учесть, чтобы нейрон был не удален
            for (uint32_t i = 0; i < m_Neurons.size(); i++)
            {
                neuron::SoinnNeuron cur_neuron = m_Neurons.at(i);
                if (!cur_neuron.is_deleted() and i != num_neuron)
                {
                    cur_dist = m_Neurons[num_neuron].setCurPointDist(cur_neuron.getWv());
                    if (cur_dist < min_dist) 
                        min_dist = cur_dist;
                }
            }
            threshold = min_dist;
        }
        return threshold;
    }

    void Soinn::processNewPoint(const wv::Point* p)
    {
        std::pair<double, double> dist = findWinners(p);
        m_Neurons[m_NumWinner].setThreshold(EvalThreshold(m_NumWinner));
        m_Neurons[m_NumSecondWinner].setThreshold(EvalThreshold(m_NumSecondWinner));
        
        if (m_Neurons[m_NumWinner].isCreateNewNode(dist.first) || m_Neurons[m_NumSecondWinner].isCreateNewNode(dist.second))
        {
            //insert new neuron
            cont::StaticArray<double> coords(m_NumDimensions);
            for (uint32_t i = 0; i < m_NumDimensions; i++)
                coords[i] = p->getConcreteCoord(i);
        
            wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);
            m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector));            
        }
        else
        {
            //increase local error of winner
            m_Neurons[m_NumWinner].setCurPointDist(p); //recalculate distance between winner and point because it might distort via threshold estimation
            m_Neurons[m_NumWinner].updateError();  
        
            updateEdgeWinSecWin();
            incrementEdgeAgeFromWinner();
            m_Neurons[m_NumWinner].incrementLocalSignals();
            updateWeights(p);
            deleteOldEdges();
        }
    }

    void Soinn::updateEdgeWinSecWin()
    {
        m_Neurons[m_NumWinner].updateEdge(m_NumSecondWinner);
        m_Neurons[m_NumSecondWinner].updateEdge(m_NumWinner);
    }

    void Soinn::incrementEdgeAgeFromWinner()
    {
        //update edges emanant from winner
        std::vector<uint32_t> neighbours = m_Neurons[m_NumWinner].incrementEdgesAge();
        //update edges incoming into winner
        for (const uint32_t num: neighbours)
        {
            m_Neurons[num].incrementConcreteEdgeAge(m_NumWinner);
        }
    }

    void Soinn::updateWeights(const wv::Point* p)
    {
        //update winner weights
        alr::AdaptLearnRateSoinn alr_win(0, m_Neurons[m_NumWinner].localSignals()); //The first parameter isn't important for soinn training
        m_Neurons[m_NumWinner].getWv()->updateWeightVector(p, &alr_win, 0);
        
        //update winner neighbours weights
        std::vector<uint32_t> neighbours = m_Neurons[m_NumWinner].getNeighbours();
        for (uint32_t num: neighbours)
        {
            alr::AdaptLearnRateSoinn alr_neigh(0, m_Neurons[num].localSignals()); //The first parameter isn't important for soinn training
            m_Neurons[num].getWv()->updateWeightVector(p, &alr_neigh, 1);
        }
    }
    
    void Soinn::InsertConcreteNeuron(const wv::Point* p)
    {
        uint32_t size = p->getNumDimensions();
        cont::StaticArray<double> coords(size);
        for (uint32_t i = 0; i < size; i++)
            coords[i] = p->getConcreteCoord(i);

        wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);
        m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector));
    }

    void Soinn::InsertConcreteEdge(uint32_t neur1, uint32_t neur2)
    {
        m_Neurons[neur1].updateEdge(neur2);
        m_Neurons[neur2].updateEdge(neur1);    
    }
    
    void Soinn::deleteOldEdges()
    {
        for (auto& s: m_Neurons)
            s.deleteOldEdges(m_AgeMax);
    }

} //nn
