#ifndef __NEURON_ABSTRACT_NEURON__
#define __NEURON_ABSTRACT_NEURON__

#include <weight_vector/AbstractWeightVector.hpp>

namespace neuron
{
    enum NeuronType
    {
        EUCLIDEAN = 1
    };
    
    class AbstractNeuron
    {
    public:
        AbstractNeuron(wv::AbstractWeightVector *wv)
            :m_wv(wv)
        { }
        
        void initRandomValues()
        {
            m_wv->initRandomValues();
        }

        wv::AbstractWeightVector* getWv()
        {
            return m_wv;
        }

    protected:
        wv::AbstractWeightVector* m_wv;         
    };
} //neuron

#endif //__NEURON_ABSTRACT_NEURON__
