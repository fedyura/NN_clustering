#ifndef __NEURAL_NETWORK_NEURAL_GAS_HPP__
#define __NEURAL_NETWORK_NEURAL_GAS_HPP__

#include <memory>
#include <neuron/NeuralGasNeuron.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace nn
{
    enum NetworkStopCriterion
    {
        LOCAL_ERROR = 1
    };

    class NeuralGas
    {
    public:
        NeuralGas(uint32_t num_dimensions, double adaptLearnRateWinner, double adaptLearnRateNotWinner, double alpha, double betta, double age_max, uint32_t lambda, NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN);

        void train(const std::vector<wv::Point*>& points, double epsilon);

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
        
        neuron::NeuralGasNeuron getNeuron(int num) const
        {
            return m_Neurons.at(num);
        }
        

        ~NeuralGas();
    
    protected:
        //Initialize network with two points from dataset
        void initialize(const std::pair<wv::Point*, wv::Point*>& points);
        
        void findWinners(const wv::Point* p);
        void updateWeights(const wv::Point* p, const alr::AbstractAdaptLearnRate* alr);
        void incrementEdgeAgeFromWinner();
        void updateEdgeWinSecWin();
        void deleteOldEdges();
        void insertNode();
        void decreaseAllErrors();

        double getErrorOnNeuron();
    
    private:
        uint32_t m_NumDimensions;
        
        double m_AdaptLearnRateWinner;
        double m_AdaptLearnRateNotWinner;
        double m_Alpha;
        double m_Betta;
        double m_AgeMax;
        uint32_t m_Lambda;

        NetworkStopCriterion m_NetStop;        //Type of network stop criterion
        neuron::NeuronType m_NeuronType;       //Type of the neuron

        std::vector<neuron::NeuralGasNeuron> m_Neurons;

        uint32_t m_NumWinner;
        uint32_t m_NumSecondWinner;

        //return true if we need to continue training, false - otherwise
        bool trainOneEpoch(const std::vector<wv::Point*>& points, double epsilon);
    };
}

#endif
