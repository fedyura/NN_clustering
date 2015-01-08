#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <algorithm>
#include <boost/format.hpp>
#include <fstream>
#include <logger/logger.hpp>
#include <neural_network/Soinn.hpp>

namespace nn
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("Soinn");
    
    Soinn::Soinn(uint32_t num_dimensions, double alpha1, double alpha2, double alpha3, double betta, double gamma, double age_max, uint32_t lambda, double C,  NetworkStopCriterion nnit, neuron::NeuronType nt)
    : m_NumDimensions(num_dimensions)
    , m_Alpha1(alpha1)
    , m_Alpha2(alpha2)
    , m_Alpha3(alpha3)
    , m_Betta(betta)
    , m_Gamma(gamma)
    , m_AgeMax(age_max)
    , m_Lambda(lambda)
    , m_C(C)
    , m_NetStop(nnit)
    , m_NeuronType(nt)
    , m_NumWinner(0)
    , m_NumSecondWinner(0)
    , m_NumEmptyNeurons(0)
    {
        if (m_NumDimensions < 1)
            throw std::runtime_error("Can't construct neural network. The number of dimensions must be more than 0.");
    }

    Soinn::~Soinn()
    {
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
            if (!m_Neurons[i].is_deleted())
            {
                delete m_Neurons[i].getWv();
                m_Neurons[i].setDeleted();
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

    void Soinn::insertNode()
    {
        //find neuron with max local error
        double max_error = -1, cur_error = 0;
        uint32_t num_neuron_max_err = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
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
            if (m_Neurons[s].is_deleted())
                continue;
            cur_error = m_Neurons[s].error();
            if (cur_error > max_error)
            {
                max_error = cur_error;
                num_sec_neuron_max_err = s;
            }
        }
        
        //set error radius of neurons
        m_Neurons[num_neuron_max_err].calcErrorRadius();
        m_Neurons[num_sec_neuron_max_err].calcErrorRadius();

        //create new neuron
        neuron::SoinnNeuron sn(&m_Neurons[num_neuron_max_err], &m_Neurons[num_sec_neuron_max_err], m_Alpha1, m_Alpha2, m_Alpha3);

        //decrease error and number of local signals
        m_Neurons[num_neuron_max_err].changeError(m_Betta);
        m_Neurons[num_sec_neuron_max_err].changeError(m_Betta);

        m_Neurons[num_neuron_max_err].changeLocalSignals(m_Gamma);
        m_Neurons[num_sec_neuron_max_err].changeLocalSignals(m_Gamma);

        //check if insertion is successful
        if (m_Neurons[num_neuron_max_err].isInsertionSuccesfull() and 
            m_Neurons[num_sec_neuron_max_err].isInsertionSuccesfull() and
            sn.isInsertionSuccesfull())
        {
            //insertion is successfull. Add new neuron and replace edges
            m_Neurons.push_back(sn);
            
            //replace edges
            m_Neurons[num_neuron_max_err].replaceNeighbour(num_sec_neuron_max_err, m_Neurons.size() - 1);
            m_Neurons[num_sec_neuron_max_err].replaceNeighbour(num_neuron_max_err, m_Neurons.size() - 1);
            
            //add edges for new neuron
            m_Neurons[m_Neurons.size() - 1].updateEdge(num_neuron_max_err);
            m_Neurons[m_Neurons.size() - 1].updateEdge(num_sec_neuron_max_err);
        }
        else
        {
            //insertion isn't successful. Restore old parameters
            m_Neurons[num_neuron_max_err].changeError(1.0 / m_Betta);
            m_Neurons[num_sec_neuron_max_err].changeError(1.0 / m_Betta);

            m_Neurons[num_neuron_max_err].changeLocalSignals(1.0 / m_Gamma);
            m_Neurons[num_sec_neuron_max_err].changeLocalSignals(1.0 / m_Gamma);

            delete sn.getWv();
        }
    }
    
    double Soinn::calcAvgLocalSignals()
    {
        m_NumEmptyNeurons = 0;
        double sumLocalSignals = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
            {
                m_NumEmptyNeurons++;
                continue;
            }
            sumLocalSignals += m_Neurons[i].localSignals();
        }
        return sumLocalSignals / (double) (m_Neurons.size() - m_NumEmptyNeurons);
    }
    
    void Soinn::deleteNodes()
    {
        //list of neurons which we will delete
        std::vector<uint32_t> deleted_neurons;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            
            if (m_Neurons[i].getNumNeighbours() == 0)
            {
                delete m_Neurons[i].getWv();   
                m_Neurons[i].setDeleted();
            }
            else if (m_Neurons[i].getNumNeighbours() == 1)
            {
                if (m_Neurons[i].localSignals() < m_C * calcAvgLocalSignals())
                    deleted_neurons.push_back(i);
            }
        }
        //delete neurons from list
        for (uint32_t num: deleted_neurons)
            deleteNeuron(num);    
    }

    void Soinn::deleteNeuron(uint32_t number)
    { 
        //delete neuron with number
        delete m_Neurons[number].getWv();   
        m_Neurons[number].setDeleted();
        
        //delete all connections between other neurons and deleted neuron
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            m_Neurons[i].deleteConcreteNeighbour(number);    
        }    
    }

    bool Soinn::trainOneEpoch(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon)
    {
        //create vector with order of iterating by points
        std::vector<uint32_t> order;
        for (uint32_t i = 0; i < points.size(); i++)
            order.push_back(i);
        
        std::random_shuffle(order.begin(), order.end());
        uint32_t iteration = 1;
        double error_before = getErrorOnNeuron();

        for (uint32_t i = 0; i < order.size(); i++)
        {
            processNewPoint(points[order[i]].get());
            if (iteration % m_Lambda == 0)
            {
                insertNode(); 
                deleteNodes();
            }
            
            iteration++;
        }
        double error_after = getErrorOnNeuron();
        
        log_netw->info((boost::format("Error before %g. Error after %g") % error_before % error_after).str());
        log_netw->info((boost::format("Network size: %d neurons") % m_Neurons.size()).str()); 
        return (error_before - getErrorOnNeuron() > epsilon);
    }

    double Soinn::getErrorOnNeuron()
    {
        double error = 0;
        for (const auto& s: m_Neurons)
        {
            if (!s.is_deleted())
                error += s.error();
        }
        //set max error for initial network of two neurons
        if (m_Neurons.size() == 2)
            error = std::numeric_limits<double>::max();
        return error/m_Neurons.size();
    }

    void Soinn::trainNetwork(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon)
    {
        initialize(std::make_pair(points[points.size()/3].get(), points[points.size()*2/3].get()));
        uint32_t iteration = 1;
        while (trainOneEpoch(points, epsilon))
        {
            log_netw->info("----------------------------------------------------------------------");
            log_netw->info((boost::format("Iteration %d") % iteration).str());
            log_netw->info((boost::format("Network size = %d, number of empty neurons = %d") % m_Neurons.size() % m_NumEmptyNeurons).str());
            iteration++;
        }
    }
} //nn
