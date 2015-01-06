#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <container/StaticArray.hpp>
#include <neuron/SoinnNeuron.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

using namespace boost::unit_test;

namespace
{
    const uint32_t TestNumDimensions = 3;
}

BOOST_AUTO_TEST_SUITE(NeuralGasNeuron)

BOOST_AUTO_TEST_CASE(test_NeuronInsertion)
{
    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 1;
    coords[1] = 2;
    coords[2] = 3;

    wv::WeightVectorContEuclidean weightVector(coords);
    neuron::SoinnNeuron sn(&weightVector);
    sn.incrementLocalSignals();
    sn.incrementLocalSignals();
    sn.setError(1.2);
    sn.CalcErrorRadius();
    
    coords[0] = 3;
    coords[1] = -2;
    coords[2] = 7;

    wv::WeightVectorContEuclidean weightVector2(coords);
    neuron::SoinnNeuron sn2(&weightVector2);
    sn2.incrementLocalSignals();
    sn2.setError(0.8);
    sn2.CalcErrorRadius();

    neuron::SoinnNeuron sn3(&sn, &sn2, 0.5, 0.25, 0.1);

    BOOST_CHECK_EQUAL(sn3.getWv()->getConcreteCoord(0), 2);
    BOOST_CHECK_EQUAL(sn3.getWv()->getConcreteCoord(1), 0);
    BOOST_CHECK_EQUAL(sn3.getWv()->getConcreteCoord(2), 5);
    
    BOOST_CHECK_EQUAL(sn3.error(), 1.0);
    BOOST_CHECK_EQUAL(sn3.localSignals(), 1.25);
    BOOST_CHECK_EQUAL((int)(sn3.errorRadius()*100 + 0.5), 8);

    delete sn3.getWv();
}

BOOST_AUTO_TEST_SUITE_END()
