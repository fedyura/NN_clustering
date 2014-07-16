#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace wv
{
    double WeightVectorContEuclidean::calcDistance(Point* p) const
    {
        const WeightVectorContEuclidean* point = dynamic_cast<WeightVectorContEuclidean*>(p);
        uint32_t size = getNumDimensions();
        if (point == NULL)
            return WRONG_POINT_TYPE;
        if (size != point->getNumDimensions())
            return WRONG_POINT_SIZE;
        
        double dist = 0.0, diff = 0.0;
        try
        {
            for (uint32_t i = 0; i < size; i++)
            {
                diff = (getConcreteCoord(i) - point->getConcreteCoord(i));
                dist += diff * diff;            
            }
        }
        catch (std::exception& ex)
        {
            return UNDEFINED_ERROR;
        }
        
        return sqrt(dist);
    }

    bool WeightVectorContEuclidean::updateWeightVector(Point* p, const alr::AbstractAdaptLearnRate* alr)
    {
        const WeightVectorContEuclidean* point = dynamic_cast<WeightVectorContEuclidean*>(p);
        uint32_t size = getNumDimensions();
        if (point == NULL)
            return false;
        if (size != point->getNumDimensions())
            return false;
        
        double rate = alr->getLearnRate(), diff = 0;
        try
        {
            for (uint32_t i = 0; i < size; i++)
            {
                diff = (point->getConcreteCoord(i) - getConcreteCoord(i));
                m_Coords[i] += rate * diff;  
            }
        }
        catch (std::exception& ex)
        {
            return false;
        }
        return true;
    }

    void WeightVectorContEuclidean::initRandomValues()
    {
        for (uint32_t i = 0; i < getNumDimensions(); i++)
            m_Coords[i] = ((double) rand() / (RAND_MAX));
    }
} //wv
