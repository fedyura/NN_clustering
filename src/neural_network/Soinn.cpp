#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <cstdio>
#include <fstream>
#include <logger/logger.hpp>
#include <neural_network/Soinn.hpp>

namespace nn
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("Soinn");
    
    Soinn::Soinn(uint32_t num_dimensions, double alpha1, double alpha2, double alpha3, double betta, double gamma, double age_max, uint32_t lambda, double C, NetworkStopCriterion nnit, neuron::NeuronType nt)
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
        log_netw->debug("Network is successfully created");    
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
        
        switch(m_NeuronType)
        {
            case neuron::NeuronType::EUCLIDEAN:
            {
                wv::WeightVectorContEuclidean* sWeightVector1 = new wv::WeightVectorContEuclidean(coords1);
                wv::WeightVectorContEuclidean* sWeightVector2 = new wv::WeightVectorContEuclidean(coords2);
                m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector1));
                m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector2));
                break;
            }
            case neuron::NeuronType::COSINE:
            {
                wv::WeightVectorCosine* sWeightVector1 = new wv::WeightVectorCosine(coords1);
                wv::WeightVectorCosine* sWeightVector2 = new wv::WeightVectorCosine(coords2);
                m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector1));
                m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector2));
                break;
            }
            default:
            {
                throw std::runtime_error("Can't construct neural network. Incorrect neuron type.");
                break;
            }
        }
        log_netw->debug("Network is successfully initialized");
    }

    std::pair<double, double> Soinn::findWinners(const wv::Point* p)
    {
        double min_dist_main = std::numeric_limits<double>::max(), min_dist_sec = std::numeric_limits<double>::max();
        double cur_dist = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
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
        //if neuron doesn't have any neighbours find mininum of distance between this and other neurons (min between cluster distance)
        else
        {
            double min_dist = std::numeric_limits<double>::max(), cur_dist = 0;
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

    void Soinn::processNewPoint(const wv::Point* p, bool train_first_layer, double threshold_sec_layer)
    {
        std::pair<double, double> dist = findWinners(p);
        if (train_first_layer)
        {
            m_Neurons[m_NumWinner].setThreshold(EvalThreshold(m_NumWinner));
            m_Neurons[m_NumSecondWinner].setThreshold(EvalThreshold(m_NumSecondWinner));
        }
        else
        {
            m_Neurons[m_NumWinner].setThreshold(threshold_sec_layer);
            m_Neurons[m_NumSecondWinner].setThreshold(threshold_sec_layer);
        }
        
        if (m_Neurons[m_NumWinner].isCreateNewNode(dist.first) || m_Neurons[m_NumSecondWinner].isCreateNewNode(dist.second))
        {
            //insert new neuron
            cont::StaticArray<double> coords(m_NumDimensions);
            for (uint32_t i = 0; i < m_NumDimensions; i++)
                coords[i] = p->getConcreteCoord(i);
            
            switch(m_NeuronType)
            {
                case neuron::NeuronType::EUCLIDEAN:
                {
                    wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);
                    m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector));            
                    break;
                }
                case neuron::NeuronType::COSINE:
                {
                    wv::WeightVectorCosine* sWeightVector = new wv::WeightVectorCosine(coords);
                    m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector));            
                    break;
                }
                default:
                {
                    throw std::runtime_error("Can't construct neural network. Incorrect neuron type.");
                    break;
                }
            }
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

        
        switch(m_NeuronType)
        {
            case neuron::NeuronType::EUCLIDEAN:
            {
                wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);
                m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector));            
                break;
            }
            case neuron::NeuronType::COSINE:
            {
                wv::WeightVectorCosine* sWeightVector = new wv::WeightVectorCosine(coords);
                m_Neurons.push_back(neuron::SoinnNeuron(sWeightVector));            
                break;
            }
            default:
            {
                throw std::runtime_error("Can't construct neural network. Incorrect neuron type.");
                break;
            }
        }   
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
    
    void Soinn::deleteNodes(bool only_no_neighbours)
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
            else if (m_Neurons[i].getNumNeighbours() == 1 and !only_no_neighbours)
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

    bool Soinn::trainOneEpoch(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon, bool train_first_layer, double threshold_sec_layer)
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
            processNewPoint(points[order[i]].get(), train_first_layer, threshold_sec_layer);
            if (iteration % m_Lambda == 0)
            {
                log_netw->debug("iteration multiple lambda");
                insertNode(); 
                deleteNodes();
            }
            
            iteration++;
        }
        //print all neurons
        deleteNodes(true);
        SealNeuronVector();
        
        double error_after = getErrorOnNeuron();
        
        log_netw->info((boost::format("Error before %g. Error after %g") % error_before % error_after).str());
        log_netw->info((boost::format("Network size = %d, number of empty neurons = %d") % m_Neurons.size() % m_NumEmptyNeurons).str());
        return (std::abs(error_before - error_after) > epsilon);
    }

    double Soinn::getErrorOnNeuron()
    {
        double error = 0;
        uint32_t num_non_empty_neurons = 0;
        for (const auto& s: m_Neurons)
        {
            if (!s.is_deleted())
            {
                error += s.error();
                num_non_empty_neurons++;
            }
        }
        //set max error for initial network of two neurons
        if (m_Neurons.size() == 2)
        {
            error = std::numeric_limits<double>::max();
            num_non_empty_neurons = 2;
        }
        m_NumEmptyNeurons = m_Neurons.size() - num_non_empty_neurons;
        return error/num_non_empty_neurons;
    }

    void Soinn::trainNetwork(const std::vector<std::shared_ptr<wv::Point>>& points, std::vector<std::vector<uint32_t>>& result,
                             uint32_t num_iteration_first_layer, uint32_t num_iteration_second_layer)
    {
        initialize(std::make_pair(points[points.size()/3].get(), points[points.size()*2/3].get()));
        uint32_t iteration = 1;
        double epsilon = 0.005;
        //train first layer
        log_netw->info("********************Train first layer************************");
        while (iteration <= num_iteration_first_layer)
        {
            trainOneEpoch(points, epsilon, true, 0);
            log_netw->info("----------------------------------------------------------------------");
            log_netw->info((boost::format("Iteration %d") % iteration).str());
            iteration++;
        }
        std::vector<std::vector<uint32_t>> first_layer_clust;
        findClustersMCL(first_layer_clust);
        //findConnectedComponents(first_layer_clust);

        //train second layer
        log_netw->info("********************Train second layer************************");
        double threshold_sec_layer = calcThresholdSecondLayer(result);
        log_netw->info((boost::format("Threshold of second layer %d") % threshold_sec_layer).str());
        
        iteration = 1;
        while (iteration <= num_iteration_second_layer)
        {
            trainOneEpoch(points, epsilon, false, threshold_sec_layer);
            log_netw->info("----------------------------------------------------------------------");
            log_netw->info((boost::format("Iteration %d") % iteration).str());
            iteration++;
        }
        findClustersMCL(result);
        //findConnectedComponents(result);
    }

    void Soinn::exportEdgesFile(const std::string& filename) const
    {
        std::unordered_map<uint64_t, double> edges; //edge => weight. Edge is two neurons (first - the smaller 32bit and second - the older one)
        uint32_t edge_weight = 1;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            std::vector<uint32_t> neighbour = m_Neurons[i].getNeighbours();
            uint64_t key = 0;
            for (uint32_t neigh: neighbour)
            {
                key = (i < neigh) ? neigh : i;
                key = key << 32;
                key += (i < neigh) ? i : neigh;
                edges.emplace(key, edge_weight);
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
    uint32_t Soinn::findPointCluster(const wv::Point* p, const std::unordered_map<uint32_t, uint32_t>& neuron_cluster) const
    {
        double min_dist_main = std::numeric_limits<double>::max();
        double cur_dist = 0;
        uint32_t winner = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            cur_dist = m_Neurons[i].getWv()->calcDistance(p);       
            if (cur_dist < min_dist_main)
            {
                min_dist_main = cur_dist;
                winner = i;
            }
        }
        //log_netw->debug((boost::format("winner = %d cluster = %d") % winner % neuron_cluster.at(winner)).str());
        return neuron_cluster.at(winner);
    }

    void Soinn::SealNeuronVector()
    {
        std::vector<neuron::SoinnNeuron> newNeurons;
        uint32_t count_notnull = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            if (i == count_notnull)
            {
                count_notnull++;
                continue;
            }
            //newNeurons.push_back(m_Neurons[i]);
            for (uint32_t j = 0; j < m_Neurons.size(); j++)
            {
                if (m_Neurons[j].is_deleted())
                    continue;
                try
                {
                    m_Neurons[j].replaceNeighbour(i, count_notnull, true);    
                }
                catch (std::runtime_error)
                {
                }
            }
            count_notnull++;    
        }
        
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            newNeurons.push_back(m_Neurons[i]);
        }
        m_Neurons = newNeurons;        
    }

    double Soinn::calcInnerClusterDistance() const
    {
        double overall_dist = 0;
        uint32_t num_edges = 0;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            for (uint32_t num: m_Neurons[i].getNeighbours())
            {
                overall_dist += m_Neurons[i].getWv()->calcDistance(m_Neurons[num].getWv());
                num_edges++;
            }
        }
        return overall_dist / (double) num_edges;
    }

    //Depth-first search
    void Soinn::dfs(cont::StaticArray<bool>& marked, int vert_number, std::vector<uint32_t>& concr_comp) const
    {
        marked[vert_number] = true;
        concr_comp.push_back(vert_number);
        for (uint32_t num: m_Neurons[vert_number].getNeighbours())
        {
            if (!marked[num])
            {
                dfs(marked, num, concr_comp);
            }
        }
    }
    
    void Soinn::findConnectedComponents(std::vector<std::vector<uint32_t>>& conn_comp) const //comp_number => vertex in component
    {
        uint32_t graph_vertex_num = m_Neurons.size();
        cont::StaticArray<bool> marked(graph_vertex_num);

        for (uint32_t i = 0; i < graph_vertex_num; i++)
            marked[i] = false;

        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (!marked[i])
            {
                std::vector<uint32_t> concr_comp;
                dfs(marked, i, concr_comp);
                conn_comp.push_back(concr_comp);
            }
        }
    }

    //Calculate between cluster distanse and sort it
    void Soinn::calcBetweenClustersDistanceVector(const std::vector<std::vector<uint32_t>>& conn_comp, std::vector<double>& dist) const
    {
        for (uint32_t i = 0; i < conn_comp.size(); i++)
            for (uint32_t j = i + 1; j < conn_comp.size(); j++)
                dist.push_back(calcDistanceBetweenTwoClusters(conn_comp[i], conn_comp[j]));
        std::sort(dist.begin(), dist.end());        
    }

    double Soinn::calcDistanceBetweenTwoClusters(std::vector<uint32_t> cluster1, std::vector<uint32_t> cluster2) const
    {
        double min_dist = std::numeric_limits<double>::max();
        for (uint32_t i = 0; i < cluster1.size(); i++)
            for (uint32_t j = 0; j < cluster2.size(); j++)
            {
                double cur_dist = m_Neurons[cluster1[i]].getWv()->calcDistance(m_Neurons[cluster2[j]].getWv());
                if (cur_dist < min_dist)
                    min_dist = cur_dist;
            }
        return min_dist;    
    }

    double Soinn::calcThresholdSecondLayer(const std::vector<std::vector<uint32_t>>& clusters) const
    {
        double inner_cluster_dist = calcInnerClusterDistance();
        double second_layer_threshold = 0;

        std::vector<double> between_cluster_dist;
        calcBetweenClustersDistanceVector(clusters, between_cluster_dist);
        
        for (uint32_t i = 0; i < between_cluster_dist.size(); i++)
        {
            if (between_cluster_dist[i] >= inner_cluster_dist)
            {
                second_layer_threshold = between_cluster_dist[i];
                break; 
            }
        }
        if (second_layer_threshold == 0)
            second_layer_threshold = inner_cluster_dist;
        return second_layer_threshold;    
    }

    void Soinn::findClustersMCL(std::vector<std::vector<uint32_t>>& clusters) const
    {
        //prepare input data for mcl
        exportEdgesFile(output_src_mcl);
    
        //run mcl algorithm
        std::string output_mcl = "mcl_clusters.tmp";    
        std::string options = " -I 2.0 --abc -o ";
        std::string command = mcl_path + output_src_mcl + options + output_mcl;
        system(command.c_str());
        
        //read output mcl file and form results
        std::ifstream fi(output_mcl);
        if (not fi)
            throw std::runtime_error("Can't read mcl output file. File " + output_mcl + " is not found");
        
        std::string line;
        while (std::getline(fi, line) and !line.empty())
        {
            std::vector<std::string> items;
            boost::split(items, line, boost::is_any_of("\t"));
            std::vector<uint32_t> one_cluster;
            for (const std::string& s: items)
            {
                uint32_t neuron_num = std::stoul(s);
                one_cluster.push_back(neuron_num);
            }
            clusters.push_back(one_cluster);
        }

        //remove temporary files
        std::remove(output_src_mcl.c_str());
        std::remove(output_mcl.c_str());
    }
} //nn
