#include <exception>
#include <cstring>
#include <typeinfo>
#include <weight_vector/WeightVectorCosine.hpp>

namespace wv
{
    double WeightVectorCosine::calcDistance(const Point* p) const
    {
        uint32_t size = getNumDimensions();
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::runtime_error("Error in function WeightVectorCosine::calcDistance. type_id of point = " + std::string(typeid(*p).name()) + " not equal with type_id of this = " + std::string(typeid(*this).name()));
        if (size != p->getNumDimensions())
            throw std::runtime_error("Wrong number of point dimension"); 
        
        double inner_product = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (uint32_t i = 0; i < size; i++)
        {
            inner_product += getConcreteCoord(i) * p->getConcreteCoord(i);
            norm1 += getConcreteCoord(i) * getConcreteCoord(i);
            norm2 += p->getConcreteCoord(i) * p->getConcreteCoord(i);
        }            
        return (1 - inner_product / sqrt(norm1 * norm2));
    }

    //distance - distance between neuron winner and point p
    void WeightVectorCosine::updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr, double distance)
    {
        uint32_t size = getNumDimensions();
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::runtime_error("Error in function WeightVectorCosine::calcDistance. type_id of point = " + std::string(typeid(*p).name()) + " not equal with type_id of this = " + std::string(typeid(*this).name()));
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

    void WeightVectorCosine::eraseOffset()
    {
        for (uint32_t i = 0; i < m_Offset.size(); i++)
            m_Offset[i] = 0;
    }
    
    double WeightVectorCosine::getOffsetValue() const 
    {
        double value = 0;
        for (uint32_t i = 0; i < m_Offset.size(); i++)
            value += m_Offset[i] * m_Offset[i];
        return value;    
    }

    Point* WeightVectorCosine::getMiddlePoints(const Point* p) const
    {
        uint32_t size = getNumDimensions();
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::bad_typeid();
        if (size != p->getNumDimensions())
            throw std::runtime_error("Wrong number of point dimension");

        cont::StaticArray<double> coords(size);
        for (uint32_t i = 0; i < size; i++)
            coords[i] = (p->getConcreteCoord(i) + getConcreteCoord(i)) / 2.0;
        
        return new wv::WeightVectorCosine(coords);        
    }
} //wv
