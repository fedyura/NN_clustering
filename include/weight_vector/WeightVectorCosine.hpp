#ifndef __WEIGHT_VECTOR_WEIGHT_VECTOR_COSINE__
#define __WEIGHT_VECTOR_WEIGHT_VECTOR_COSINE__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <container/StaticArray.hpp>
#include <weight_vector/AbstractWeightVector.hpp>

namespace wv //wv => weight_vector
{
    class WeightVectorCosine: public AbstractWeightVector
    {
    public:
        uint32_t getNumDimensions() const
        {
            return m_Coords.size();
        }

        double getConcreteCoord(uint32_t number) const
        {
            return m_Coords[number]; //m_Coords.at(number); 
        }
        
        //Calculate distance between this weight vector and Point p
        virtual double calcDistance(const Point* p) const;
       
        //Update weight vector in the one training iteration. 
        virtual void updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr, double distance);

        virtual void eraseOffset();
        virtual double getOffsetValue() const;

        //Calculate point in the middle between two points and return it
        //This function allocate memory which it is needed to delete
        virtual Point* getMiddlePoints(const Point* p) const;
        
        explicit WeightVectorCosine(const cont::StaticArray<double>& coords)
            : m_Coords(coords)
            , m_Offset(coords)
        {
            for (uint32_t i = 0; i < m_Offset.size(); i++)
                m_Offset[i] = 0;
        }
   
    private:
        cont::StaticArray<double> m_Coords;
        cont::StaticArray<double> m_Offset; //coord offset during the one iteration
    };
} //wv

#endif //__WEIGHT_VECTOR_WEIGHT_VECTOR_COSINE__
