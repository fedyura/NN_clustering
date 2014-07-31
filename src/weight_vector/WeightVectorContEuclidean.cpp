#include <exception>
#include <cstring>
#include <typeinfo>
#include <weight_vector/WeightVectorContEuclidean.hpp>

#include <iostream>

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
            throw std::exception();
        }
        
        return sqrt(dist);
    }

    //distance - distance between neuron winner and point p
    void WeightVectorContEuclidean::updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr, double distance)
    {
        uint32_t size = getNumDimensions();
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::bad_typeid();
        if (size != p->getNumDimensions())
            throw std::runtime_error("Wrong number of point dimension"); 
        
        double rate = alr->getLearnRate(distance), diff = 0;
        try
        {
            for (uint32_t i = 0; i < size; i++)
            {
                diff = (p->getConcreteCoord(i) - getConcreteCoord(i));
                m_Coords[i] += rate * diff;
                m_Offset[i] += rate * diff;  
            }
        }
        catch (std::exception& ex)
        {
            throw std::exception();
        }
    }

    void WeightVectorContEuclidean::eraseOffset()
    {
        for (uint32_t i = 0; i < m_Offset.size(); i++)
            m_Offset[i] = 0;
    }
    
    double WeightVectorContEuclidean::getOffsetValue() const 
    {
        double value = 0;
        for (uint32_t i = 0; i < m_Offset.size(); i++)
            value += m_Offset[i] * m_Offset[i];
        return value;    
    }
} //wv
