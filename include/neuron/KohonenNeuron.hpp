#ifndef __NEURON_KOHONEN_NEURON_HPP__
#define __NEURON_KOHONEN_NEURON_HPP__

#include <neuron/AbstractNeuron.hpp>
#include <memory>

namespace neuron
{
    class KohonenNeuron: public AbstractNeuron
    {
    public:
        KohonenNeuron(/*std::shared_ptr<wv::AbstractWeightVector>& sAwv*/ wv::AbstractWeightVector* wv)
            :AbstractNeuron(wv)
        { }
        KohonenNeuron()
            :AbstractNeuron(NULL)
        { }    
                
    };
} //neuron
#endif //__NEURON_KOHONEN_NEURON_HPP__
