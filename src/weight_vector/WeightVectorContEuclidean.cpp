#include <exception>
#include <cstring>
#include <typeinfo>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace wv
{
    double WeightVectorContEuclidean::calcDistance(const Point* p) const
    {
        uint32_t size = getNumDimensions();
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::bad_typeid();
        if (size != p->getNumDimensions())
            throw std::runtime_error("Wrong number of point dimension"); 
        
        double dist = 0.0, diff = 0.0;
        try
        {
            for (uint32_t i = 0; i < size; i++)
            {
                diff = (getConcreteCoord(i) - p->getConcreteCoord(i));
                dist += diff * diff;            
            }
        }
        catch (std::exception& ex)
        {
            return UNDEFINED_ERROR;
        }
        
        return sqrt(dist);
    }

    int WeightVectorContEuclidean::updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr)
    {
        uint32_t size = getNumDimensions();
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::bad_typeid();
        if (size != p->getNumDimensions())
            throw std::runtime_error("Wrong number of point dimension"); 
        
        double rate = alr->getLearnRate(), diff = 0;
        try
        {
            for (uint32_t i = 0; i < size; i++)
            {
                diff = (p->getConcreteCoord(i) - getConcreteCoord(i));
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
