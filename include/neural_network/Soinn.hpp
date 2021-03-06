#ifndef __NEURAL_NETWORK_SOINN_HPP__
#define __NEURAL_NETWORK_SOINN_HPP__

#include <memory>
#include <neural_network/Common.hpp>
#include <neuron/SoinnNeuron.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>
#include <weight_vector/WeightVectorCosine.hpp>

namespace nn
{
    const std::string mcl_path = "/home/yura/local/bin/mcl ";
    const std::string output_src_mcl = "soinn_output_mcl.tmp";
    
    class Soinn
    {
      public:
        Soinn(uint32_t num_dimensions, double alpha1, double alpha2, double alpha3, double betta, double gamma, double age_max, uint32_t lambda, double C, NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN);    

        void trainNetwork(const std::vector<std::shared_ptr<wv::Point>>& points, std::vector<std::vector<uint32_t>>& result,
                          uint32_t num_iteration_first_layer, uint32_t num_iteration_sec_layer);
        void exportEdgesFile(const std::string& filename) const;
        uint32_t findPointCluster(const wv::Point* p, const std::unordered_map<uint32_t, uint32_t>& neuron_cluster) const;
        void findConnectedComponents(std::vector<std::vector<uint32_t>>& conn_comp) const;
        
        //functions for testing
        double getNeuronError(uint32_t i) const
        {
            return m_Neurons.at(i).error();
        } 

        double getNeuronCoord(uint32_t num, uint32_t coord) const
        {
            return m_Neurons.at(num).getWv()->getConcreteCoord(coord);
        }

        double getWinner() const
        {
            return m_NumWinner;
        }

        double getSecWinner() const
        {
            return m_NumSecondWinner;
        }
        
        neuron::SoinnNeuron getNeuron(int num) const
        {
            return m_Neurons.at(num);
        }

        void incrementLocalSignals(int num)
        {
            m_Neurons.at(num).incrementLocalSignals();
        }

        void setError(int num, double error)
        {
            m_Neurons.at(num).setError(error);
        }

        uint32_t numEmptyNeurons()
        {
            return m_NumEmptyNeurons;
        }
        
        ~Soinn();
        
      protected:
        //Initialize network with two points from dataset
        void initialize(const std::pair<wv::Point*, wv::Point*>& points);

        std::pair<double, double> findWinners(const wv::Point* p);
        double EvalThreshold(uint32_t num_neuron);
        void processNewPoint(const wv::Point* p, bool train_first_layer = true, double threshold_sec_layer = 0);
        void updateEdgeWinSecWin();
        void incrementEdgeAgeFromWinner();
        void updateWeights(const wv::Point* p);
        void deleteOldEdges();

        void insertNode();
        void deleteNodes(bool only_no_neighbours = false);
        
        double calcAvgLocalSignals();
        void deleteNeuron(uint32_t number);
        double getErrorOnNeuron();

        double calcInnerClusterDistance() const;
        void calcBetweenClustersDistanceVector(const std::vector<std::vector<uint32_t>>& conn_comp, std::vector<double>& dist) const;
        double calcThresholdSecondLayer(const std::vector<std::vector<uint32_t>>& clusters) const;

        void findClustersMCL(std::vector<std::vector<uint32_t>>& clusters) const;
        
        //for tests
        void InsertConcreteNeuron(const wv::Point* p); 
        void InsertConcreteEdge(uint32_t neur1, uint32_t neur2);
      
      private:
        //return true if we need to continue training, false - otherwise
        bool trainOneEpoch(const std::vector<std::shared_ptr<wv::Point>>& points, double epsilon, bool train_first_layer = true, double threshold_sec_layer = 0);
        void SealNeuronVector();
        void dfs(cont::StaticArray<bool>& marked, int vert_number, std::vector<uint32_t>& concr_comp) const;
        double calcDistanceBetweenTwoClusters(std::vector<uint32_t> cluster1, std::vector<uint32_t> cluster2) const;
        
        uint32_t m_NumDimensions;
        
        double m_Alpha1;
        double m_Alpha2;
        double m_Alpha3;
        double m_Betta;
        double m_Gamma;
        
        double m_AgeMax;
        uint32_t m_Lambda;

        double m_C;
        
        NetworkStopCriterion m_NetStop;        //Type of network stop criterion
        neuron::NeuronType m_NeuronType;       //Type of the neuron

        std::vector<neuron::SoinnNeuron> m_Neurons;

        uint32_t m_NumWinner;
        uint32_t m_NumSecondWinner;
        
        uint32_t m_NumEmptyNeurons;        
    };
}
#endif //__NEURAL_NETWORK_SOINN_HPP__
