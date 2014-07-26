#ifndef __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_KOHONEN_SCHEMA__
#define __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_KOHONEN_SCHEMA__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <cmath> 

namespace alr //alr => adapt_learn_rate
{
    class KohonenParameters
    {        
    public:
        KohonenParameters(double sigmaBeg, double tSigma, double rateBeg, double tEnd)
            : m_SigmaBeg(sigmaBeg)
            , m_TSigma(tSigma)
            , m_RateBeg(rateBeg)
            , m_TEnd(tEnd)
        { }

        double sigmaBeg() const
        {
            return m_SigmaBeg;
        }

        double tSigma() const
        {
            return m_TSigma;
        }

        double rateBeg() const
        {
            return m_RateBeg;
        }

        double tEnd() const
        {
            return m_TEnd;
        }

    private:
        double m_SigmaBeg;
        double m_TSigma;
        double m_RateBeg;
        double m_TEnd;        
    };
    
    class AdaptLearnRateKohonenSchema: public AbstractAdaptLearnRate
    {
    public:
        virtual double getLearnRate(double distance) const
        {
            return getNeighbourCoeff(m_IterNumber, distance)*getLearningRateCoeff(m_IterNumber);
        }

        AdaptLearnRateKohonenSchema(uint32_t iterNumber, KohonenParameters kp)
            : AbstractAdaptLearnRate(iterNumber)
            , m_Kp(kp)
        { }
    
    private:
        KohonenParameters m_Kp;
        
        double sigmaFunction(uint32_t iter_number) const //sigma(t) = sigma_0 * exp(-t/tau_sigma) 
        {
            return m_Kp.sigmaBeg()*exp(-(double)iter_number/m_Kp.tSigma());
        }
        
        double getLearningRateCoeff(uint32_t iter_number) const //a(t) = a_0 * exp(-t/tau_a)
        {
            return m_Kp.rateBeg()*exp(-(double)iter_number/m_Kp.tEnd());
        }

        double getNeighbourCoeff(uint32_t iter_number, double distance) const //n(t) = exp(-d*d/2*sigma(t)*sigma(t))
        {
            return exp(-distance*distance/(2 * sigmaFunction(iter_number)*sigmaFunction(iter_number)));
        }
    };
} //alr

#endif //__ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_KOHONEN_SCHEMA__
