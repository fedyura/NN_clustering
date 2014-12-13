#include <adapt_learn_rate/AdaptLearnRateNeuralGas.hpp>
#include <algorithm>
#include <boost/format.hpp>
#include <fstream>
#include <logger/logger.hpp>
#include <neural_network/NeuralGas.hpp>

namespace nn
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("NeuralGas");
    
    NeuralGas::NeuralGas(uint32_t num_dimensions, double adaptLearnRateWinner, double adaptLearnRateNotWinner, double alpha, double betta, double age_max, uint32_t lambda, NetworkStopCriterion nnit, neuron::NeuronType nt)
    : m_NumDimensions(num_dimensions)
    , m_AdaptLearnRateWinner(adaptLearnRateNotWinner)
    , m_AdaptLearnRateNotWinner(adaptLearnRateNotWinner)
    , m_Alpha(alpha)
    , m_Betta(betta)
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

    void NeuralGas::initialize(const std::pair<wv::Point*, wv::Point*>& points)
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
        m_Neurons.push_back(neuron::NeuralGasNeuron(sWeightVector1));
        m_Neurons.push_back(neuron::NeuralGasNeuron(sWeightVector2));
        m_Neurons[0].updateEdge(1);
        m_Neurons[1].updateEdge(0);
    }

    NeuralGas::~NeuralGas()
    {
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
            if (m_Neurons[i].getWv() != NULL)
            {
                delete m_Neurons[i].getWv();
                m_Neurons[i].setZeroPointer();
            }
    }
    
    void NeuralGas::findWinners(const wv::Point* p)
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
        //increase local error of winner
        m_Neurons[m_NumWinner].updateError();
    }
    
    void NeuralGas::updateWeights(const wv::Point* p, const alr::AbstractAdaptLearnRate* alr)
    {
        m_Neurons[m_NumWinner].getWv()->updateWeightVector(p, alr, 0);
        m_Neurons[m_NumSecondWinner].getWv()->updateWeightVector(p, alr, 1); //not null distance
    }

    void NeuralGas::incrementEdgeAgeFromWinner()
    {
        //update edges emanant from winner
        std::vector<uint32_t> neighbours = m_Neurons[m_NumWinner].incrementEdgesAge();
        //update edges incoming into winner
        for (const uint32_t num: neighbours)
        {
            m_Neurons[num].incrementConcreteEdgeAge(m_NumWinner);
        }
    }

    void NeuralGas::updateEdgeWinSecWin()
    {
        m_Neurons[m_NumWinner].updateEdge(m_NumSecondWinner);
        m_Neurons[m_NumSecondWinner].updateEdge(m_NumWinner);
    }

    void NeuralGas::deleteOldEdges()
    {
        for (auto& s: m_Neurons)
            s.deleteOldEdges(m_AgeMax);
    }

    void NeuralGas::insertNode()
    {
        //find neuron with max local error
        double max_error = -1, cur_error = 0;
        uint32_t num_neuron_max_err = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            cur_error = m_Neurons[i].error();
            if (cur_error > max_error)
            {
                max_error = cur_error;
                num_neuron_max_err = i;
            }
        }
        
        //find its neighbour with max local error
        std::vector<uint32_t> neighbours = m_Neurons[num_neuron_max_err].getNeighbours();
        max_error = -1, cur_error = 0;
        uint32_t num_sec_neuron_max_err = 0;
        for (const uint32_t s: neighbours)
        {
            cur_error = m_Neurons[s].error();
            if (cur_error > max_error)
            {
                max_error = cur_error;
                num_sec_neuron_max_err = s;
            }
        }

        //create new node between two neurons
        cont::StaticArray<double> coords(m_NumDimensions);
        for (uint32_t i = 0; i < m_NumDimensions; i++)
            coords[i] = (m_Neurons[num_sec_neuron_max_err].getWv()->getConcreteCoord(i) + m_Neurons[num_neuron_max_err].getWv()->getConcreteCoord(i)) / 2;
        
        wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);
        m_Neurons.push_back(neuron::NeuralGasNeuron(sWeightVector));
        
        //replace edges
        m_Neurons[num_neuron_max_err].replaceNeighbour(num_sec_neuron_max_err, m_Neurons.size() - 1);
        m_Neurons[num_sec_neuron_max_err].replaceNeighbour(num_neuron_max_err, m_Neurons.size() - 1);
        
        //add edges for new neuron
        m_Neurons[m_Neurons.size() - 1].updateEdge(num_neuron_max_err);
        m_Neurons[m_Neurons.size() - 1].updateEdge(num_sec_neuron_max_err);

        //set new error values
        m_Neurons[num_neuron_max_err].changeError(m_Alpha);
        m_Neurons[num_sec_neuron_max_err].changeError(m_Alpha);
        m_Neurons[m_Neurons.size() - 1].setError(m_Neurons[num_neuron_max_err].error());
    }

    void NeuralGas::decreaseAllErrors()
    {
        for (auto& s: m_Neurons)
            s.changeError(m_Betta);
    }

    bool NeuralGas::trainOneEpoch(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon)
    {
        //create vector with order of iterating by points
        std::vector<uint32_t> order;
        for (uint32_t i = 0; i < points.size(); i++)
            order.push_back(i);
        
        std::random_shuffle(order.begin(), order.end());
        uint32_t iteration = 1;
        alr::AdaptLearnRateNeuralGas alrn(0, m_AdaptLearnRateWinner, m_AdaptLearnRateNotWinner);
        double error_before = getErrorOnNeuron();

        for (uint32_t i = 0; i < order.size(); i++)
        {
            findWinners(points[order[i]].get());
            updateWeights(points[order[i]].get(), &alrn);
            incrementEdgeAgeFromWinner();
            updateEdgeWinSecWin();
            deleteOldEdges();
            
            if (iteration % m_Lambda == 0)
            {
                insertNode(); 
            }
            
            decreaseAllErrors();
            iteration++;
        }
        double error_after = getErrorOnNeuron();
        
        log_netw->info((boost::format("Error before %g. Error after %g") % error_before % error_after).str());
        log_netw->info((boost::format("Network size: %d neurons") % m_Neurons.size()).str()); 
        return (error_before - getErrorOnNeuron() > epsilon);
    }

    double NeuralGas::getErrorOnNeuron()
    {
        double error = 0;
        for (const auto& s: m_Neurons)
            error += s.error();
        //set max error for initial network of two neurons
        if (m_Neurons.size() == 2)
            error = std::numeric_limits<double>::max()*2;
        return error/m_Neurons.size();
    }

    void NeuralGas::trainNetwork(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon)
    {
        initialize(std::make_pair(points[points.size()/3].get(), points[points.size()*2/3].get()));
        uint32_t iteration = 1;
        while (trainOneEpoch(points, epsilon))
        {
            log_netw->info("----------------------------------------------------------------------");
            log_netw->info((boost::format("Iteration %d") % iteration).str());
            iteration++;
        }
    }

    void NeuralGas::exportEdgesFile(const std::string& filename) const
    {
        std::unordered_map<uint64_t, double> edges; //edge => weight. Edge is two neurons (first - the smaller 32bit and second - the older one)
        uint32_t edge_weight = 1;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            std::vector<uint32_t> neighbour = m_Neurons[i].getNeighbours();
            uint64_t key = 0;
            for (uint32_t neigh: neighbour)
            {
                key = (i < neigh) ? neigh : i;
                key = key << 32;
                key += (i < neigh) ? i : neigh;
                edges.emplace(key, 1);
            }
        }

        //print edges in file
        std::ofstream out(filename, std::ios::out);
        for (const auto s: edges)
        {
            uint64_t key = s.first; 
            out << (key & 0x00000000FFFFFFFF) << " ";
            key = key >> 32;
            out << (key & 0x00000000FFFFFFFF) << " " << edge_weight << std::endl;
        }

        out.close();
    }

    //find cluster for each point for built network
    uint32_t NeuralGas::findPointCluster(const wv::Point* p, const std::unordered_map<uint32_t, uint32_t>& neuron_cluster) const
    {
        double min_dist_main = std::numeric_limits<double>::max();
        double cur_dist = 0;
        uint32_t winner = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            cur_dist = m_Neurons[i].getWv()->calcDistance(p);       
            if (cur_dist < min_dist_main)
            {
                min_dist_main = cur_dist;
                winner = i;
            }
        }
        return neuron_cluster.at(winner);
    }
}
