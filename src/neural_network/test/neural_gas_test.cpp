#include <adapt_learn_rate/AdaptLearnRateNeuralGas.hpp>
#include <boost/test/unit_test.hpp>
#include <neural_network/NeuralGas.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

using namespace boost::unit_test;
using namespace nn;

namespace 
{
    const uint32_t TestNumDimensions = 3;
    const double TestAdaptLearnRateWinner = 0.5;
    const double TestAdaptLearnRateSecWinner = 0.1;
    const double TestAlpha = 0.7;
    const double TestBetta = 0.9;
    const double TestAgeMax = 0;
    const uint32_t TestLambda = 2;
}

class TestNeuralGas: public NeuralGas
{
public:
    TestNeuralGas(NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN)
        : NeuralGas(TestNumDimensions, TestAdaptLearnRateWinner, TestAdaptLearnRateSecWinner, TestAlpha, TestBetta, TestAgeMax, TestLambda, nnit, nt)
    { }    
    
    void initializeNetwork()
    {
        cont::StaticArray<double> coords1(TestNumDimensions);
        coords1[0] = 1;
        coords1[1] = 2;
        coords1[2] = 3;
        
        cont::StaticArray<double> coords2(TestNumDimensions);
        coords2[0] = 3;
        coords2[1] = 4;
        coords2[2] = 5;

        wv::WeightVectorContEuclidean wv1(coords1);
        wv::WeightVectorContEuclidean wv2(coords2);
        
        initialize(std::make_pair(&wv1, &wv2));
        insertNode();
    }
        
    void testFindWinners(const wv::Point* p)
    {
        findWinners(p);
    }
    
    void testUpdateWeights(const wv::Point* p, const alr::AbstractAdaptLearnRate* alr)
    {
        updateWeights(p, alr);
    }
    
    void testIncrementEdgeAgeFromWinner()
    {
        incrementEdgeAgeFromWinner();
    }
    
    void testUpdateEdgeWinSecWin()
    {
        updateEdgeWinSecWin();
    }
    
    void testDeleteOldEdges()
    {
        deleteOldEdges();
    }
    
    void testInsertNode()
    {
        insertNode();
    }
    
    void testDecreaseAllErrors()
    {
        decreaseAllErrors();
    }

    double testGetErrorOnNeuron()
    {
        return getErrorOnNeuron();
    }
};

BOOST_AUTO_TEST_SUITE(TestNeuralGasFunctions)

BOOST_AUTO_TEST_CASE(test_InitializeNN)
{
    TestNeuralGas tng;
    tng.initializeNetwork();

    BOOST_CHECK_EQUAL(tng.getNeuronCoord(0, 0), 1);
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(0, 1), 2);
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(0, 2), 3);
    
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(1, 0), 3);
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(1, 1), 4);
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(1, 2), 5);
    
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(2, 0), 2);
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(2, 1), 3);
    BOOST_CHECK_EQUAL(tng.getNeuronCoord(2, 2), 4);
    
    BOOST_CHECK_EQUAL(tng.getNeuronError(0), 0);
    BOOST_CHECK_EQUAL(tng.getNeuronError(1), 0);
    BOOST_CHECK_EQUAL(tng.getNeuronError(2), 0);
}

BOOST_AUTO_TEST_CASE(test_findWinners)
{
    TestNeuralGas tng;
    tng.initializeNetwork();

    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 2.2;
    coords[1] = 3.2;
    coords[2] = 4.2;

    wv::WeightVectorContEuclidean wv(coords);
    tng.testFindWinners(&wv);

    BOOST_CHECK_EQUAL(tng.getWinner(), 2);
    BOOST_CHECK_EQUAL(tng.getSecWinner(), 1);
    
    BOOST_CHECK_EQUAL(tng.getNeuronError(0), 0);
    BOOST_CHECK_EQUAL(tng.getNeuronError(1), 0);
    BOOST_CHECK_EQUAL(int(tng.getNeuronError(2)*1000), 120);
}

BOOST_AUTO_TEST_CASE(test_updateWeights)
{
    TestNeuralGas tng;
    tng.initializeNetwork();
    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 2.2;
    coords[1] = 3.2;
    coords[2] = 4.2;

    wv::WeightVectorContEuclidean wv(coords);
    tng.testFindWinners(&wv);
    
    alr::AdaptLearnRateNeuralGas alrn(0, 0.2, 0.1);
    tng.testUpdateWeights(&wv, &alrn);
    
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(2, 0)*100), 204);
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(2, 1)*100), 304);
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(2, 2)*100), 404);
    
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(1, 0)*100 + 0.5), 292);
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(1, 1)*100 + 0.5), 392);
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(1, 2)*100 + 0.5), 492);
}

BOOST_AUTO_TEST_CASE(test_EdgesTransform)
{
    TestNeuralGas tng;
    tng.initializeNetwork();
    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 2.2;
    coords[1] = 3.2;
    coords[2] = 4.2;

    wv::WeightVectorContEuclidean wv(coords);
    tng.testFindWinners(&wv);
    
    alr::AdaptLearnRateNeuralGas alrn(0, 0.2, 0.1);
    tng.testUpdateWeights(&wv, &alrn);
    
    tng.testIncrementEdgeAgeFromWinner();
    std::vector<uint32_t> nb = tng.getNeuron(2).getNeighbours();
    
    BOOST_REQUIRE_EQUAL(nb.size(), 2);
    std::sort(nb.begin(), nb.end());
    BOOST_CHECK_EQUAL(nb[0], 0);
    BOOST_CHECK_EQUAL(nb[1], 1);  
    
    nb = tng.getNeuron(1).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 1);
    
    tng.testUpdateEdgeWinSecWin();
    tng.testDeleteOldEdges();    
    nb = tng.getNeuron(2).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 1);
    BOOST_CHECK_EQUAL(nb[0], 1);
    
    nb = tng.getNeuron(1).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 1);
    BOOST_CHECK_EQUAL(nb[0], 2);
    
    nb = tng.getNeuron(0).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 0);

    //Insert node
    tng.testInsertNode();
    
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(3, 0)*100 + 0.51), 248);
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(3, 1)*100 + 0.51), 348);
    BOOST_CHECK_EQUAL(int(tng.getNeuronCoord(3, 2)*100 + 0.5), 448);

    nb = tng.getNeuron(1).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 1);
    BOOST_CHECK_EQUAL(nb[0], 3);
    
    nb = tng.getNeuron(3).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 2);
    std::sort(nb.begin(), nb.end());
    BOOST_CHECK_EQUAL(nb[0], 1);
    BOOST_CHECK_EQUAL(nb[1], 2);

    nb = tng.getNeuron(2).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 1);
    std::sort(nb.begin(), nb.end());
    BOOST_CHECK_EQUAL(nb[0], 3);

    nb = tng.getNeuron(0).getNeighbours();
    BOOST_REQUIRE_EQUAL(nb.size(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
