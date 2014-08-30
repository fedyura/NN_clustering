#ifndef __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_MARTINES__
#define __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_MARTINES__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <cmath>

namespace alr //alr => adapt_learn_rate
{
    class AdaptLearnRateNeuralGas: public AbstractAdaptLearnRate
    {
    public:
        virtual double getLearnRate(double distance) const
        {
            return (distance == 0) ? m_AdaptWinner : m_AdaptNotWinner;
        }
        
        AdaptLearnRateNeuralGas(uint32_t iterNumber, double adaptWinner, double adaptNotWinner)
            : AbstractAdaptLearnRate(iterNumber)
            , m_AdaptWinner(adaptWinner)
            , m_AdaptNotWinner(adaptNotWinner)
            { } 
    private:
        double m_AdaptWinner;          //adaptation coefficient for winner (e_w)
        double m_AdaptNotWinner;       //adaptation coefficient for not winner (e_n)
    };
} //alr

#endif //__ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_MARTINES__
