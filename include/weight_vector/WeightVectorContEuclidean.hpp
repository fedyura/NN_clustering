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
       
        //Update weight vector in the one training iteration. Return false if error
        virtual int updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr);

        //Initialize weight vector with random values
        virtual void initRandomValues();

        explicit WeightVectorContEuclidean(const cont::StaticArray<double>& coords)
            : m_Coords(coords)
        { }
   
        //~WeightVectorContEuclidean();  //if dynamic array instead of vector
    private:
        cont::StaticArray<double> m_Coords;
    };
} //wv

#endif //__WEIGHT_VECTOR_WEIGHT_VECTOR_CONT_EUCLIDEAN__

