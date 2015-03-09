#include <cmath>
#include <quality_measures/quality_measures.hpp>

#include <iostream>

namespace qm
{
    QualityMeasures::QualityMeasures(const std::vector<std::string>& classes, const std::vector<uint32_t> clusters)
      : m_N(classes.size())
      , m_TP(0)
      , m_TN(0)
      , m_FP(0)
      , m_FN(0)
    {
        if (classes.size() != clusters.size())
            throw std::runtime_error("Wrong data is passed in QualityMeasures class. QualityMeasures object can't be created");
        
        for (uint32_t i = 0; i < classes.size(); i++)
        {
            m_FreqClasses[classes[i]]++;
            m_FreqClusters[clusters[i]]++;
            m_ClData[clusters[i]][classes[i]]++;
        }
    }
    
    double QualityMeasures::evalPurity() const
    {
        uint32_t num_matching_classes = 0;
        for (const auto& cl: m_ClData)
        {
            uint32_t max_samples_from_one_class = 0;
            for (const auto& cl_num: cl.second)
            {
                if (cl_num.second > max_samples_from_one_class)
                    max_samples_from_one_class = cl_num.second;
            }
            num_matching_classes += max_samples_from_one_class;
        }

        return (double) num_matching_classes / m_N;
    }
    
    double QualityMeasures::evalMutualInformation() const
    {
        double mutualInform = 0;
        for (const auto& cl: m_ClData)
        {
            for (const auto& cl_num: cl.second)
            {
                mutualInform += cl_num.second * log2((double) m_N * cl_num.second / (m_FreqClusters.at(cl.first) * m_FreqClasses.at(cl_num.first))) / m_N; 
            }
        }
        return mutualInform;
    }

    template <typename T>
    double QualityMeasures::evalEntropy(const std::map<T, uint32_t>& data) const
    {
        double entropy = 0;
        for (const auto& freq: data)
        {
            if (freq.second == 0)
                continue;
            entropy += freq.second * log2 ((double) freq.second / m_N) / m_N;    
        }
        return -entropy;
    }

    double QualityMeasures::evalNormalizedMutualInformation() const
    {
        return evalMutualInformation() * 2 / (evalEntropy(m_FreqClusters) + evalEntropy(m_FreqClasses));
    }

    void QualityMeasures::findStatisticalQualityMeasures()
    {
        uint32_t TP_FP = 0;
        uint32_t num_events_cluster = 0;
        ClusterData::const_iterator it_clust, it_clust_inner;
        
        for (it_clust = m_ClData.begin(); it_clust != m_ClData.end(); it_clust++)
        {
            num_events_cluster = 0;
            for (const auto& classes: it_clust->second)
            {
                it_clust_inner = it_clust;
                it_clust_inner++;    
                for (; it_clust_inner != m_ClData.end(); it_clust_inner++)
                {
                    std::map<std::string, uint32_t>::const_iterator it = it_clust_inner->second.find(classes.first);
                    if (it != it_clust_inner->second.end())
                        m_FN += classes.second * it->second;
                }                
                m_TP += classes.second * (classes.second - 1) / 2;
                num_events_cluster += classes.second;
            }
            
            TP_FP += num_events_cluster * (num_events_cluster - 1) / 2;
        }
        
        m_FP = TP_FP - m_TP;
        m_TN = m_N * (m_N - 1) / 2 - m_FP - m_FN - m_TP;
    }
}
