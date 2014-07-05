#ifndef __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_KOHONEN_SCHEMA__
#define __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_KOHONEN_SCHEMA__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <iostream>

namespace alr //alr => adapt_learn_rate
{
    class AdaptLearnRateKohonenSchema: public AbstractAdaptLearnRate
    {
    public:
        virtual double getLearnRate() const
        {
            return getNeighbourCoeff(m_IterNumber, m_Distance)*getLearningRateCoeff(m_IterNumber);
        }

        AdaptLearnRateKohonenSchema(uint32_t iterNumber, double distance, double sigmaBeg,
                                    double tSigma, double rateBeg, double tEnd)
            : AbstractAdaptLearnRate(iterNumber, distance)
            , m_SigmaBeg(sigmaBeg)
            , m_TSigma(tSigma)
            , m_RateBeg(rateBeg)
            , m_TEnd(tEnd)
        { }
    
    private:
        double m_SigmaBeg; //sigma_0
        double m_TSigma;   //tau_sigma
        double m_RateBeg;  //a_0
        double m_TEnd;     //tau_a
        
        double sigmaFunction(uint32_t iter_number) const //sigma(t) = sigma_0 * exp(-t/tau_sigma) 
        {
            return m_SigmaBeg*exp(-(double)iter_number/m_TSigma);
        }
        
        double getLearningRateCoeff(uint32_t iter_number) const //a(t) = a_0 * exp(-t/tau_a)
        {
            return m_RateBeg*exp(-(double)iter_number/m_TEnd);
        }

        double getNeighbourCoeff(uint32_t iter_number, double distance) const //n(t) = exp(-d*d/2*sigma(t)*sigma(t))
        {
            return exp(-distance*distance/(2 * sigmaFunction(iter_number)*sigmaFunction(iter_number)));
        }
    };
} //alr

#endif //__ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_KOHONEN_SCHEMA__
