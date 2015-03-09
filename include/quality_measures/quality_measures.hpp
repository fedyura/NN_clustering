#ifndef __QUALITY_MEASURES_QUALITY_MEASURES_HPP__
#define __QUALITY_MEASURES_QUALITY_MEASURES_HPP__

#include <map>
#include <string>
#include <vector>
#include <unordered_map>

namespace qm
{
    typedef std::map<uint32_t, std::map<std::string, uint32_t>> ClusterData; //cluster_id => (class_id => quantity)
    
    class QualityMeasures
    {
      public:
        QualityMeasures(const std::vector<std::string>& classes, const std::vector<uint32_t> clusters);
        
        double evalPurity() const;          
        double evalNormalizedMutualInformation() const;
        void findStatisticalQualityMeasures();
        
        uint32_t TP() { return m_TP; }
        uint32_t TN() { return m_TN; }
        uint32_t FP() { return m_FP; }
        uint32_t FN() { return m_FN; }
        
        double randIndex() 
        {
            uint32_t denominator = m_TP + m_TN + m_FP + m_FN;
            if (denominator == 0)
                throw std::runtime_error("Statistical measures aren't initialized");
            return (double)(m_TP + m_TN) / denominator;
        }

        double precision()
        {
            uint32_t denominator = m_TP + m_FP;
            if (denominator == 0)
                throw std::runtime_error("Statistical measures aren't initialized");
            return (double)(m_TP) / denominator;
        }

        double recall()
        {
            uint32_t denominator = m_TP + m_FN;
            if (denominator == 0)
                throw std::runtime_error("Statistical measures aren't initialized");
            return (double)(m_TP) / denominator;
        }

        double Fscore(double betta)
        {
            double nominator = (betta*betta + 1)*precision()*recall();
            double denominator = betta*betta*precision() + recall();
            if (denominator == 0)
                throw std::runtime_error("Statistical measures aren't initialized");
            return nominator / denominator;    

        }

      private:
        double evalMutualInformation() const;
        
        template <typename T>
        double evalEntropy(const std::map<T, uint32_t>& data) const;
        
        uint32_t m_N;
        uint32_t m_TP, m_TN, m_FP, m_FN;
        ClusterData m_ClData;
        std::map<std::string, uint32_t> m_FreqClasses; //class name => number of occurences
        std::map<uint32_t, uint32_t> m_FreqClusters; //cluster id => number of occurences
    };
}

#endif //__QUALITY_MEASURES_QUALITY_MEASURES_HPP__
