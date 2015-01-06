#ifndef __NEURAL_NETWORK_SOINN_HPP__
#define __NEURAL_NETWORK_SOINN_HPP__

#include <memory>
#include <neural_network/Common.hpp>
#include <neuron/SoinnNeuron.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace nn
{
    class Soinn
    {
      public:
        Soinn(uint32_t num_dimensions, double alpha1, double alpha2, double alpha3, double betta, double gamma, double age_max, uint32_t lambda, NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN);    
        
      protected:
        //Initialize network with two points from dataset
        void initialize(const std::pair<wv::Point*, wv::Point*>& points);

        std::pair<double, double> findWinners(const wv::Point* p);
        double EvalThreshold(uint32_t num_neuron);
        void processNewPoint(const wv::Point* p);
        void updateEdgeWinSecWin();
        void incrementEdgeAgeFromWinner();
        void updateWeights(const wv::Point* p);
        void deleteOldEdges(); 
      
      private:
        uint32_t m_NumDimensions;
        
        double m_Alpha1;
        double m_Alpha2;
        double m_Alpha3;
        double m_Betta;
        double m_Gamma;
        
        double m_AgeMax;
        uint32_t m_Lambda;
        
        NetworkStopCriterion m_NetStop;        //Type of network stop criterion
        neuron::NeuronType m_NeuronType;       //Type of the neuron

        std::vector<neuron::SoinnNeuron> m_Neurons;

        uint32_t m_NumWinner;
        uint32_t m_NumSecondWinner;
        
        uint32_t m_NumEmptyNeurons;        
    };
}
#endif //__NEURAL_NETWORK_SOINN_HPP__
