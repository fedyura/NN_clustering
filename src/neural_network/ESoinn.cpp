#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <cstdio>
#include <fstream>
#include <logger/logger.hpp>
#include <neural_network/ESoinn.hpp>
#include <unordered_set>

namespace nn
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("ESoinn");
    
    ESoinn::ESoinn(uint32_t num_dimensions, double age_max, uint32_t lambda, double C1, double C2, NetworkStopCriterion nnit, neuron::NeuronType nt)
    : m_NumDimensions(num_dimensions)
    , m_AgeMax(age_max)
    , m_Lambda(lambda)
    , m_C1(C1)
    , m_C2(C2)
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

    ESoinn::~ESoinn()
    {
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
            if (!m_Neurons[i].is_deleted())
            {
                delete m_Neurons[i].getWv();
                m_Neurons[i].setDeleted();
            }
    }
    
    void ESoinn::initialize(const std::pair<wv::Point*, wv::Point*>& points, const std::pair<std::string, std::string>& src_label)
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
        m_Neurons.push_back(neuron::ESoinnNeuron(sWeightVector1, src_label.first));
        m_Neurons.push_back(neuron::ESoinnNeuron(sWeightVector2, src_label.second));
        log_netw->debug("Network is successfully initialized");
    }

    std::pair<double, double> ESoinn::findWinners(const wv::Point* p)
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

    double ESoinn::evalThreshold(uint32_t num_neuron)
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
                neuron::ESoinnNeuron cur_neuron = m_Neurons.at(i);
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

    void ESoinn::processNewPoint(const wv::Point* p, std::string label)
    {
        std::pair<double, double> dist = findWinners(p);
        double winner_threshold = evalThreshold(m_NumWinner);
        double sec_winner_threshold = evalThreshold(m_NumSecondWinner);

        if (dist.first > winner_threshold || dist.second > sec_winner_threshold)
        {
            //insert new neuron
            cont::StaticArray<double> coords(m_NumDimensions);
            for (uint32_t i = 0; i < m_NumDimensions; i++)
                coords[i] = p->getConcreteCoord(i);
            
            wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);
            m_Neurons.push_back(neuron::ESoinnNeuron(sWeightVector, label));            
        }
        else
        {
            incrementEdgeAgeFromWinner();
            updateEdgeWinSecWin();
            m_Neurons[m_NumWinner].incrementLocalSignals();            
            m_Neurons[m_NumWinner].evalAndSetDensity(midDistNeighbours(m_NumWinner));
            
            updateWeights(p);
            deleteOldEdges();
        }
    }
    
    void ESoinn::incrementEdgeAgeFromWinner()
    {
        //update edges emanant from winner
        std::vector<uint32_t> neighbours = m_Neurons[m_NumWinner].incrementEdgesAge();
        //update edges incoming into winner
        for (const uint32_t num: neighbours)
        {
            m_Neurons[num].incrementConcreteEdgeAge(m_NumWinner);
        }
    }
    
    void ESoinn::updateEdgeWinSecWin()
    {
        //create edge if although one of neurons hasn't class label or if their class labels are equal
        if (m_Neurons[m_NumWinner].curClass() == 0 || m_Neurons[m_NumSecondWinner].curClass() == 0 ||
            m_Neurons[m_NumWinner].curClass() == m_Neurons[m_NumSecondWinner].curClass())
        {
            m_Neurons[m_NumWinner].updateEdge(m_NumSecondWinner);
            m_Neurons[m_NumSecondWinner].updateEdge(m_NumWinner);
        }
        //else delete edge
        else
        {
            m_Neurons[m_NumWinner].deleteConcreteNeighbour(m_NumSecondWinner);
            m_Neurons[m_NumSecondWinner].deleteConcreteNeighbour(m_NumWinner);
        }        
    }

    double ESoinn::midDistNeighbours(uint32_t num_neuron)
    {
        std::vector<uint32_t> neighbours = m_Neurons.at(num_neuron).getNeighbours();
        double sum_dist = 0;
        for (uint32_t num: neighbours)
            sum_dist += m_Neurons[num_neuron].setCurPointDist(m_Neurons.at(num).getWv());
        return (sum_dist == 0) ? 0 : sum_dist / (double) neighbours.size();    
    }

    void ESoinn::updateWeights(const wv::Point* p)
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

    void ESoinn::deleteOldEdges()
    {
        for (auto& s: m_Neurons)
            s.deleteOldEdges(m_AgeMax);
    }
    
    bool ESoinn::isNodeAxis(uint32_t num_neuron)
    {
        std::vector<uint32_t> neighbours = m_Neurons.at(num_neuron).getNeighbours();
        double cur_density = m_Neurons[num_neuron].density();
        double max_density = 0;
        uint32_t neuron_max_density = 0;
        for (uint32_t num: neighbours)
        {
            if (max_density < m_Neurons[num].density())
            {
                max_density = m_Neurons[num].density();
                neuron_max_density = num;
            }
        }
        if (max_density > cur_density)
        {
            m_Neurons[num_neuron].setNeighbourMaxDensity(neuron_max_density);
            return false;
        }
        m_Neurons[num_neuron].setNeighbourMaxDensity(num_neuron);
        return true;
    }
    
    uint32_t ESoinn::findAxisForNode(uint32_t number)
    {
        uint32_t cur_neuron = number;
        uint32_t max_density_neighbour = m_Neurons[cur_neuron].neighbourMaxDensity();
        while(cur_neuron != max_density_neighbour)
        {
            cur_neuron = max_density_neighbour;
            max_density_neighbour = m_Neurons[cur_neuron].neighbourMaxDensity();            
        }
        return cur_neuron;
    }
    
    double ESoinn::meanDensity(uint32_t class_number) const
    {
        uint32_t num_neurons = 0;
        double sum = 0;
        for (const auto& neur: m_Neurons)
        {
            if (neur.is_deleted())
                continue;
            
            if (neur.curClass() == class_number)
            {
                num_neurons++;
                sum += neur.density();
            }
        }

        return sum / (double) num_neurons;        
    }
    
    double ESoinn::getAlpha(double maxDensity, double meanDensity) const
    {
        double alpha = 0;
        if (maxDensity < 3 * meanDensity && maxDensity >= 2 * meanDensity)
            alpha = 0.5;
        else if (maxDensity >= 3 * meanDensity)
            alpha = 1.0;
        return alpha;         
    }
    
    void ESoinn::labelClasses(std::map<uint32_t, uint32_t>& node_axis)
    {
        //find neighbour with max density for each neuron
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            
            isNodeAxis(i);
        }
        
        //find axis for each node
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            
            uint32_t num_axis = findAxisForNode(i);
            node_axis.emplace(i, num_axis);
            m_Neurons[i].setCurClass(num_axis);
        }
    }

    //mergeClasses: node => set of neighbours
    void ESoinn::findMergedClasses(const std::map<uint32_t, uint32_t>& node_axis, std::map<uint32_t, std::set<uint32_t>>& mergeClasses) const
    {
        //iterate all edges
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            
            std::vector<uint32_t> neighbours = m_Neurons.at(i).getNeighbours();
            for (uint32_t num: neighbours)
            {
                if (num <= i)
                    continue;
                
                if (m_Neurons.at(i).curClass() != m_Neurons.at(num).curClass())
                {
                    const uint32_t A_axes = node_axis.at(i);
                    const uint32_t B_axes = node_axis.at(num);
                    assert(A_axes != B_axes);
                        
                    const double Amax = m_Neurons.at(A_axes).density();
                    const double Bmax = m_Neurons.at(B_axes).density();
                    const double min_density = std::min(m_Neurons.at(i).density(), m_Neurons.at(num).density());
                    if (min_density > getAlpha(Amax, meanDensity(A_axes)) * Amax ||
                        min_density > getAlpha(Bmax, meanDensity(B_axes)) * Bmax)
                    {
                        //Add two classes as need to merge 
                        mergeClasses[A_axes].emplace(B_axes);
                        mergeClasses[B_axes].emplace(A_axes);
                    }
                }
            }
        }
    }

    //Depth-first search
    void dfs(std::unordered_set<uint32_t>& marked, uint32_t vert_number,
             std::set<uint32_t>& concr_comp, const std::map<uint32_t, std::set<uint32_t>>& src)
    {
        marked.emplace(vert_number);
        concr_comp.emplace(vert_number);
        for (uint32_t num: src.at(vert_number))
        {
            if (marked.count(num) == 0)
            {
                dfs(marked, num, concr_comp, src);
            }
        }
    }
    
    void ESoinn::findAxesMapping(const std::map<uint32_t, std::set<uint32_t>>& src, std::map<uint32_t, uint32_t>& mapping) const
    {
        std::unordered_set<uint32_t> marked_axes;
        std::vector<std::set<uint32_t>> conn_comp;

        for (const auto& it: src)
        {
            if (marked_axes.count(it.first) == 0)
            {
                std::set<uint32_t> concr_comp;
                dfs(marked_axes, it.first, concr_comp, src);
                conn_comp.push_back(concr_comp);
            }
        }

        for (const auto& it: conn_comp)
        {
            const uint32_t min_vertex = *(it.begin());
            for (const uint32_t& vertex: it)
                mapping.emplace(vertex, min_vertex);
        }
    }
    
    void ESoinn::updateClassLabels()
    {
        std::map<uint32_t, uint32_t> node_axis; //node => axis
        labelClasses(node_axis);

        //find classes (numbers of axis) which we need to merge
        std::map<uint32_t, std::set<uint32_t>> mergeClasses;
        findMergedClasses(node_axis, mergeClasses); 
        
        //mapping for neuron classes
        std::map<uint32_t, uint32_t> mapping;
        findAxesMapping(mergeClasses, mapping);

        //merge classes
        for (auto& neur: m_Neurons)
        {
            if (neur.is_deleted())
                continue; 
            
            const std::map<uint32_t, uint32_t>::const_iterator it = mapping.find(neur.curClass());
            if (it != mapping.end())
                neur.setCurClass(it->second);
        }
    }

    void ESoinn::deleteEdgesDiffClasses()
    {
        std::map<uint32_t, uint32_t> mergeClasses;
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            
            std::vector<uint32_t> neighbours = m_Neurons.at(i).getNeighbours();
            for (uint32_t num: neighbours)
            {
                if (m_Neurons.at(i).curClass() != m_Neurons.at(num).curClass())
                    m_Neurons.at(i).deleteConcreteNeighbour(num);
            }
        }
    }

    double ESoinn::calcAvgDensity()
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
            sumLocalSignals += m_Neurons[i].density();
        }
        return sumLocalSignals / (double) (m_Neurons.size() - m_NumEmptyNeurons);
    }

    void ESoinn::deleteNodes()
    {
        //list of neurons which we will delete
        std::vector<uint32_t> deleted_neurons;
        //double avg_density = calcAvgDensity();
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            
            if (m_Neurons[i].getNumNeighbours() == 0)
            {
                delete m_Neurons[i].getWv();   
                m_Neurons[i].setDeleted();
            }
            
            if (m_Neurons[i].getNumNeighbours() == 1)
            {
                if (m_Neurons[i].density() < m_C1 * calcAvgDensity())
                    deleted_neurons.push_back(i);
            }
            else if (m_Neurons[i].getNumNeighbours() == 2)
            {
                if (m_Neurons[i].density() < m_C2 * calcAvgDensity())
                    deleted_neurons.push_back(i);
            }
        }
        //delete neurons from list
        for (uint32_t num: deleted_neurons)
            deleteNeuron(num);    

        //delete neurons without neighbours (it may appeared after deletion other neurons)
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;
            
            if (m_Neurons[i].getNumNeighbours() == 0)
            {
                delete m_Neurons[i].getWv();   
                m_Neurons[i].setDeleted();
            }
        }
    }
    
    void ESoinn::deleteNeuron(uint32_t number)
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
    
    void ESoinn::trainOneEpoch(const std::vector<std::shared_ptr<wv::Point>>& points, const std::vector<std::string>& labels)
    {
        //create vector with order of iterating by points
        std::vector<uint32_t> order;
        for (uint32_t i = 0; i < points.size(); i++)
            order.push_back(i);
        
        std::random_shuffle(order.begin(), order.end());
        uint32_t iteration = 1;

        for (uint32_t i = 0; i < order.size(); i++)
        {
            processNewPoint(points[order[i]].get(), (labels.empty()) ? "0" : labels[order[i]]);
            if (iteration % m_Lambda == 0)
            {
                log_netw->debug("iteration multiple lambda");
                updateClassLabels();
                deleteEdgesDiffClasses();
                deleteNodes();
            }
            
            iteration++;
        }
        deleteNodes();
        SealNeuronVector();
        
        log_netw->info((boost::format("Network size = %d, number of empty neurons = %d") % m_Neurons.size() % m_NumEmptyNeurons).str());
    }

    void ESoinn::trainNetwork(const std::vector<std::shared_ptr<wv::Point>>& points, std::vector<std::vector<uint32_t>>& result,
                              const std::vector<std::string>& labels, uint32_t num_iteration_first_layer)
    {
        const uint32_t first_neuron_index = points.size()/3;
        const uint32_t sec_neuron_index = points.size()*2/3;
        initialize(std::make_pair(points[first_neuron_index].get(), points[sec_neuron_index].get()), std::make_pair(labels[first_neuron_index], labels[sec_neuron_index]));
        uint32_t iteration = 1;
        //train first layer
        log_netw->info("********************Train first layer************************");
        while (iteration <= num_iteration_first_layer)
        {
            trainOneEpoch(points, labels);
            log_netw->info("----------------------------------------------------------------------");
            log_netw->info((boost::format("Iteration %d") % iteration).str());
            iteration++;
        }
        findClustersMCL(result);
    }
    
    void ESoinn::trainNetworkNoiseReduction(const std::vector<std::shared_ptr<wv::Point>>& points, const std::vector<std::string>& labels,
                                            uint32_t num_iteration_first_layer)
    {
        if (points.size() != labels.size())
            throw std::runtime_error("Size of labels vector differs from size of points vector");
        const uint32_t first_neuron_index = points.size()/3;
        const uint32_t sec_neuron_index = points.size()*2/3;
        initialize(std::make_pair(points[first_neuron_index].get(), points[sec_neuron_index].get()), std::make_pair(labels[first_neuron_index], labels[sec_neuron_index]));
        uint32_t iteration = 1;
        //train first layer
        log_netw->info("********************Train first layer************************");
        while (iteration <= num_iteration_first_layer)
        {
            trainOneEpoch(points, labels);
            log_netw->info("----------------------------------------------------------------------");
            log_netw->info((boost::format("Iteration %d") % iteration).str());
            iteration++;
        }
    }
    
    void ESoinn::exportEdgesFile(const std::string& filename) const
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
    uint32_t ESoinn::findPointCluster(const wv::Point* p, const std::unordered_map<uint32_t, uint32_t>& neuron_cluster) const
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

    void ESoinn::SealNeuronVector()
    {
        std::vector<neuron::ESoinnNeuron> newNeurons;
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

    void ESoinn::findClustersMCL(std::vector<std::vector<uint32_t>>& clusters) const
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
        //std::remove(output_src_mcl.c_str());
        //std::remove(output_mcl.c_str());
    }
    
    void ESoinn::dumpNetwork() const
    {
        for (uint32_t i = 0; i < m_Neurons.size(); i++)
        {
            if (m_Neurons[i].is_deleted())
                continue;

            log_netw->debug((boost::format("neuron = %d:") % i).str());
            for (const uint32_t num: m_Neurons[i].getNeighbours())
            {
                log_netw->debug((boost::format(" %d") % num).str());
            }
        }
    }

    void ESoinn::printNetworkNodesFile(const std::string& filename) const
    {
        std::ofstream of(filename);
        for (const auto& neur: m_Neurons)
        {
            if (neur.is_deleted())
                continue;
            
            const wv::AbstractWeightVector* av = neur.getWv();
            for (uint32_t j = 0; j < av->getNumDimensions(); j++)
            {
                of << av->getConcreteCoord(j) << ",";
            }
            of << neur.label() << std::endl;
        }

        of.close();
    }

    void ESoinn::printNetworkClustersFile(const std::string& filename, const std::vector<std::vector<uint32_t>>& clusters, uint32_t start_value) const
    {
        std::ofstream of(filename);
        for (uint32_t i = 0; i < clusters.size(); i++) //  const auto& clust: clusters)
        {
            if (clusters[i].size() < 5)
                continue;
            for (const uint32_t neuron_num: clusters[i])
            {
                const neuron::ESoinnNeuron& neur = m_Neurons[neuron_num];
                if (neur.is_deleted())
                    continue;
            
                const wv::AbstractWeightVector* av = neur.getWv();
                for (uint32_t j = 0; j < av->getNumDimensions(); j++)
                {
                    of << av->getConcreteCoord(j) << ",";
                }
                of << i + start_value << std::endl;      
            }
        }
        
        of.close();
    }

    //Special functions for tests
    void ESoinn::InsertConcreteNeuron(const wv::Point* p, const std::string& neur_class)
    {
        uint32_t size = p->getNumDimensions();
        cont::StaticArray<double> coords(size);
        for (uint32_t i = 0; i < size; i++)
            coords[i] = p->getConcreteCoord(i);

        wv::WeightVectorContEuclidean* sWeightVector = new wv::WeightVectorContEuclidean(coords);
        m_Neurons.push_back(neuron::ESoinnNeuron(sWeightVector, neur_class));
    }

    void ESoinn::InsertConcreteEdge(uint32_t neur1, uint32_t neur2)
    {
        m_Neurons[neur1].updateEdge(neur2);
        m_Neurons[neur2].updateEdge(neur1);    
    }
} //nn
