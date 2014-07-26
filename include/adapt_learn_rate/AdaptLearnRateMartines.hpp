#ifndef __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_MARTINES__
#define __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_MARTINES__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <cmath>

namespace alr //alr => adapt_learn_rate
{
    class AdaptLearnRateMartines: public AbstractAdaptLearnRate
    {
    public:
        virtual double getLearnRate(double distance) const
        {
            return m_RateBegin*pow(m_RateFinal/m_RateBegin, (double)m_IterNumber/(double)m_MaxIterNumber);
        }
        
        AdaptLearnRateMartines(uint32_t iterNumber, uint32_t maxItemNumber,
                               double rateBegin, double rateFinal)
            : AbstractAdaptLearnRate(iterNumber)
            , m_MaxIterNumber(maxItemNumber)
            , m_RateBegin(rateBegin)
            , m_RateFinal(rateFinal)
            { } 
    private:
        uint32_t m_MaxIterNumber; //max number of iteration (t_max)
        double m_RateBegin;       //initial rate of learning (e_i)
        double m_RateFinal;       //final rate of learning (e_f)
    };
} //alr

#endif //__ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_MARTINES__
