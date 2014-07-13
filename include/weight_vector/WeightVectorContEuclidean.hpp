#ifndef __WEIGHT_VECTOR_WEIGHT_VECTOR_CONT_EUCLIDEAN__
#define __WEIGHT_VECTOR_WEIGHT_VECTOR_CONT_EUCLIDEAN__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <container/StaticArray.hpp>
#include <vector>
#include <weight_vector/AbstractWeightVector.hpp>

namespace wv //wv => weight_vector
{
    class WeightVectorContEuclidean: public AbstractWeightVector
    {
    public:
        uint32_t getNumDimensions() const
        {
            return m_Coords.size();
        }

        double getConcreteCoord(uint32_t number) const
        {
            return m_Coords.at(number);
        }
        
        //Calculate distance between this weight vector and Point p
        virtual double calcDistance(Point* p) const;
       
        //Update weight vector in the one training iteration. Return false if error
        virtual bool updateWeightVector(Point* p, const alr::AbstractAdaptLearnRate* alr);

        explicit WeightVectorContEuclidean(const cont::StaticArray<double>& coords)
            : m_Coords(coords)
        { }
   
        //~WeightVectorContEuclidean();  //if dynamic array instead of vector
    private:
        cont::StaticArray<double> m_Coords;
    };

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
} //wv

#endif //__WEIGHT_VECTOR_WEIGHT_VECTOR_CONT_EUCLIDEAN__

