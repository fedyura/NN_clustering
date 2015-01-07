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
        Soinn(uint32_t num_dimensions, double alpha1, double alpha2, double alpha3, double betta, double gamma, double age_max, uint32_t lambda, double C, NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN);    

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
        
        ~Soinn();
        
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

        void insertNode();
        void deleteNodes();
        
        //for tests
        void InsertConcreteNeuron(const wv::Point* p); 
        void InsertConcreteEdge(uint32_t neur1, uint32_t neur2);
      
      private:
        double calcAvgLocalSignals();
        void deleteNeuron(uint32_t number);
        
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
