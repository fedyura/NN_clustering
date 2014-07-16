#include <boost/test/unit_test.hpp>
#include <neural_network/KohonenNN.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(KohonenNN)

BOOST_AUTO_TEST_CASE(test_InitializeNN)
{
    nn::KohonenNN knn1(5, 3);
    
    for (uint32_t j = 0; j < 5; j++)
    {
        wv::AbstractWeightVector* awv = knn1.getNeuron(j).getWv();
        wv::WeightVectorContEuclidean* wv = dynamic_cast<wv::WeightVectorContEuclidean*>(awv);
        BOOST_REQUIRE(wv != NULL);

        BOOST_CHECK(wv->getNumDimensions() == 3);
        for (uint32_t i = 0; i < wv->getNumDimensions(); i++)
            BOOST_CHECK(wv->getConcreteCoord(i) >= 0 and wv->getConcreteCoord(i) <= 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
