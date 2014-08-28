#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <container/StaticArray.hpp>
#include <neuron/NeuralGasNeuron.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

using namespace boost::unit_test;

namespace
{
    const uint32_t TestNumDimensions = 3;
}

BOOST_AUTO_TEST_SUITE(NeuralGasNeuron)

BOOST_AUTO_TEST_CASE(test_NeuronNeighbours)
{
    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 1;
    coords[1] = 2;
    coords[2] = 3;

    wv::WeightVectorContEuclidean weightVector(coords);
    neuron::NeuralGasNeuron ngn(&weightVector);

    //Add 3 neighbours
    ngn.updateEdge(1);
    ngn.updateEdge(2);
    ngn.updateEdge(3);

    //Check neighbours
    std::vector<uint32_t> neighbours = ngn.getNeighbours();
    std::sort(neighbours.begin(), neighbours.end());

    BOOST_CHECK_EQUAL(neighbours[0], 1);
    BOOST_CHECK_EQUAL(neighbours[1], 2);
    BOOST_CHECK_EQUAL(neighbours[2], 3);

    neighbours = ngn.incrementEdgesAge();
    std::sort(neighbours.begin(), neighbours.end());

    BOOST_CHECK_EQUAL(neighbours[0], 1);
    BOOST_CHECK_EQUAL(neighbours[1], 2);
    BOOST_CHECK_EQUAL(neighbours[2], 3);

    //Increment age of edges and delete old edges. Edge 1 must be deleted
    BOOST_CHECK_THROW(ngn.incrementConcreteEdgeAge(7), std::out_of_range);
    ngn.incrementConcreteEdgeAge(1);
    ngn.deleteOldEdges(1);
    
    neighbours = ngn.incrementEdgesAge();
    std::sort(neighbours.begin(), neighbours.end());
    
    BOOST_CHECK_EQUAL(neighbours[0], 2);
    BOOST_CHECK_EQUAL(neighbours[1], 3);

    //Replace edge 2 onto 6. Age must be saved
    ngn.incrementConcreteEdgeAge(2);
    ngn.replaceNeighbour(2, 6);
    neighbours = ngn.incrementEdgesAge();
    std::sort(neighbours.begin(), neighbours.end());
    
    BOOST_CHECK_EQUAL(neighbours[0], 3);
    BOOST_CHECK_EQUAL(neighbours[1], 6);
    
    BOOST_CHECK_THROW(ngn.replaceNeighbour(2, 4), std::runtime_error); 
    ngn.incrementConcreteEdgeAge(6);
    
    //Edge 6 must be deleted
    ngn.deleteOldEdges(4);
    neighbours = ngn.incrementEdgesAge();
    BOOST_CHECK_EQUAL(neighbours[0], 3);
}

BOOST_AUTO_TEST_CASE(test_NeuronError)
{
    cont::StaticArray<double> coords(TestNumDimensions);

    wv::WeightVectorContEuclidean weightVector(coords);
    neuron::NeuralGasNeuron ngn(&weightVector);
    
    ngn.setError(10);
    BOOST_CHECK_EQUAL(ngn.error(), 10);
    ngn.changeError(0.8);
    BOOST_CHECK_EQUAL(ngn.error(), 8);
}

BOOST_AUTO_TEST_SUITE_END()
