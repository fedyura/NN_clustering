#ifndef __NEURON_SOINN_NEURON_HPP__
#define __NEURON_SOINN_NEURON_HPP__

#include <assert.h>
#include <neuron/NeuralGasNeuron.hpp>

namespace neuron
{
    class SoinnNeuron: public NeuralGasNeuron
    {
      public:
        SoinnNeuron(wv::AbstractWeightVector* wv)
          : NeuralGasNeuron(wv)
          , m_Threshold(std::numeric_limits<double>::max())
          , m_LocalSignals(1)
          , m_ErrorRadius(0)
          , m_IsDeleted(false)
        { }
            
        SoinnNeuron()
          : NeuralGasNeuron(NULL)
          , m_Threshold(std::numeric_limits<double>::max())
          , m_LocalSignals(1)
          , m_ErrorRadius(0)
          , m_IsDeleted(false)
        { }

        //This conctructor allocate memory so memory has to be deleted
        SoinnNeuron(const SoinnNeuron* neur1, const SoinnNeuron* neur2, double alpha1, double alpha2, double alpha3)
          : NeuralGasNeuron(neur1->getWv()->getMiddlePoints(neur2->getWv()), alpha1*(neur1->error() + neur2->error()))
          , m_LocalSignals(alpha2 * (neur1->localSignals() + neur2->localSignals()))
          , m_ErrorRadius(alpha3 * (neur1->errorRadius() + neur2->errorRadius()))  
          , m_IsDeleted(false)
        { }
        
        bool isCreateNewNode(double distance) const
        {
            return distance > m_Threshold;
        }

        void incrementLocalSignals()
        {
            m_LocalSignals++;
        }

        double localSignals() const
        {
            return m_LocalSignals;
        }

        double calcErrorRadius()
        {
            assert(m_LocalSignals != 0);
            m_ErrorRadius = error() / m_LocalSignals;
            return m_ErrorRadius;
        }

        double errorRadius() const
        {
            return m_ErrorRadius;
        }

        void changeLocalSignals(double koeff)
        {
            m_LocalSignals = koeff * m_LocalSignals;
        }
        
        bool isInsertionSuccesfull() const
        {
            return (error() / localSignals() <= m_ErrorRadius);
        }

        void setThreshold(double threshold)
        {
            m_Threshold = threshold;
        }

        void setDeleted()
        {
            m_IsDeleted = true;
        }

        bool is_deleted() const
        {
            return m_IsDeleted;
        }

      private:
        double m_Threshold;
        double m_LocalSignals;
        double m_ErrorRadius;
        bool m_IsDeleted;    
    };
} //neuron

#endif //__NEURON_SOINN_NEURON_HPP__
