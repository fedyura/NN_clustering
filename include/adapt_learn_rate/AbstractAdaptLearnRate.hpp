#ifndef __ADAPT_LEARN_RATE_ABSTRACT_ADAPT_LEARN_RATE__
#define __ADAPT_LEARN_RATE_ABSTRACT_ADAPT_LEARN_RATE__

#include <cstdint>

namespace alr //alr => adapt_learn_rate
{
    //This class returns the learning rate for updating weights
    class AbstractAdaptLearnRate
    {
    public:
        virtual double getLearnRate() const = 0;
        
        AbstractAdaptLearnRate(uint32_t iterNumber, double distance)
            : m_IterNumber(iterNumber)
            , m_Distance(distance)
        { } 
    protected:
        uint32_t m_IterNumber; //the number of iteration
        double m_Distance;     //the distance between neuron winner and updated neuron. Distance is zero if they are matched (if the neuron winner is updated)
    };
} //alr

#endif //__ADAPT_LEARN_RATE_ABSTRACT_ADAPT_LEARN_RATE__

