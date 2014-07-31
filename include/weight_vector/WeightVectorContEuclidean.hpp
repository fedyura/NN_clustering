#ifndef __WEIGHT_VECTOR_WEIGHT_VECTOR_CONT_EUCLIDEAN__
#define __WEIGHT_VECTOR_WEIGHT_VECTOR_CONT_EUCLIDEAN__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <container/StaticArray.hpp>
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
        virtual double calcDistance(const Point* p) const;
       
        //Update weight vector in the one training iteration. 
        virtual void updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr, double distance);

        virtual void eraseOffset();
        virtual double getOffsetValue() const;
        
        explicit WeightVectorContEuclidean(const cont::StaticArray<double>& coords)
            : m_Coords(coords)
            , m_Offset(coords)
        {
            for (uint32_t i = 0; i < m_Offset.size(); i++)
                m_Offset[i] = 0;
        }
   
        //~WeightVectorContEuclidean();  //if dynamic array instead of vector
    private:
        cont::StaticArray<double> m_Coords;
        cont::StaticArray<double> m_Offset; //coord offset during the one iteration
    };
} //wv

#endif //__WEIGHT_VECTOR_WEIGHT_VECTOR_CONT_EUCLIDEAN__

