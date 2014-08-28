#ifndef __NEURON_KOHONEN_NEURON_HPP__
#define __NEURON_KOHONEN_NEURON_HPP__

#include <neuron/AbstractNeuron.hpp>

namespace neuron
{
    class KohonenNeuron: public AbstractNeuron
    {
    public:
        KohonenNeuron(wv::AbstractWeightVector* wv, double min_potential)
            : AbstractNeuron(wv)
            , m_Potential(1)
            , m_MayBeWinner(true)
            , m_MinPotential(min_potential)
        { }
        KohonenNeuron()
            : AbstractNeuron(NULL)
            , m_Potential(1)
            , m_MayBeWinner(false)
            , m_MinPotential(1)
        { }
        
        void decreasePotential()
        {
            m_Potential -= m_MinPotential;
            if (m_Potential < m_MinPotential)
                m_MayBeWinner = false;
        }
        
        void increasePotential(uint32_t num_neurons)
        {
            m_Potential += 1.0 / (double) num_neurons;
            if (m_Potential > m_MinPotential)
                m_MayBeWinner = true;
        }
        
        bool mayBeWinner() const
        {
            return m_MayBeWinner;
        }

    private:
        double m_Potential; //potential of a neuron. Influenced on neuron win  
        bool m_MayBeWinner;
        double m_MinPotential;                        
    };
} //neuron
#endif //__NEURON_KOHONEN_NEURON_HPP__
