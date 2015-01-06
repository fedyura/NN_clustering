#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <neural_network/Soinn.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

using namespace boost::unit_test;
using namespace nn;

namespace 
{
    const uint32_t TestNumDimensions = 3;
    const double TestAlpha1 = 0.2;
    const double TestAlpha2 = 0.5;
    const double TestAlpha3 = 0.7;
    const double TestBetta  = 0.6;
    const double TestGamma  = 0.8;
    const double TestAgeMax = 1;
    const uint32_t TestLambda = 2;
}

class TestSoinn: public Soinn
{
  public:
    TestSoinn(NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN)
      : Soinn(TestNumDimensions, TestAlpha1, TestAlpha2, TestAlpha3, TestBetta, TestGamma, TestAgeMax, TestLambda, nnit, nt)
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
        
        cont::StaticArray<double> coords3(TestNumDimensions);
        coords3[0] = 4;
        coords3[1] = 3;
        coords3[2] = 6;
        
        cont::StaticArray<double> coords4(TestNumDimensions);
        coords4[0] = 7;
        coords4[1] = 8;
        coords4[2] = 9;
        
        wv::WeightVectorContEuclidean wv1(coords1);
        wv::WeightVectorContEuclidean wv2(coords2);
        wv::WeightVectorContEuclidean wv3(coords3);
        wv::WeightVectorContEuclidean wv4(coords4);
        
        initialize(std::make_pair(&wv1, &wv2));
        InsertConcreteNeuron(&wv3);
        InsertConcreteNeuron(&wv4);

        InsertConcreteEdge(0, 1);
        InsertConcreteEdge(1, 3);
    }
    
    std::pair<double, double> testFindWinners(const wv::Point* p)
    {
        return findWinners(p);
    }

    double testEvalThreshold(uint32_t num_neuron)
    {
        return EvalThreshold(num_neuron);
    }

    void testUpdateWeights(const wv::Point* p)
    {
        updateWeights(p);
    }

    void testProcessNewPoint(const wv::Point* p)
    {
        processNewPoint(p);
    }
};
    
BOOST_AUTO_TEST_SUITE(TestSoinnFunctions)

BOOST_AUTO_TEST_CASE(test_InitializeNN)
{
    TestSoinn tn;
    tn.initializeNetwork();

    BOOST_CHECK_EQUAL(tn.getNeuronCoord(0, 0), 1);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(0, 1), 2);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(0, 2), 3);
    
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(1, 0), 3);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(1, 1), 4);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(1, 2), 5);
    
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(2, 0), 4);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(2, 1), 3);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(2, 2), 6);
    
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(3, 0), 7);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(3, 1), 8);
    BOOST_CHECK_EQUAL(tn.getNeuronCoord(3, 2), 9);
   
    BOOST_CHECK_EQUAL(tn.getNeuronError(0), 0);
    BOOST_CHECK_EQUAL(tn.getNeuronError(1), 0);
    BOOST_CHECK_EQUAL(tn.getNeuronError(2), 0);
    BOOST_CHECK_EQUAL(tn.getNeuronError(3), 0);
}

BOOST_AUTO_TEST_CASE(test_FindWinners)
{
    TestSoinn ts;
    ts.initializeNetwork();
    
    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 6;
    coords[1] = 6;
    coords[2] = 7;

    wv::WeightVectorContEuclidean wv(coords);
    std::pair<double, double> dist = ts.testFindWinners(&wv);
    BOOST_CHECK_EQUAL(dist.first, 3);
    BOOST_CHECK_EQUAL(int(dist.second*100+0.5), 374);

    BOOST_CHECK_EQUAL(ts.getWinner(), 3);
    BOOST_CHECK_EQUAL(ts.getSecWinner(), 2);
}

BOOST_AUTO_TEST_CASE(test_EvalThreshold)
{
    TestSoinn ts;
    ts.initializeNetwork();
    
    BOOST_CHECK_EQUAL(int(ts.testEvalThreshold(1)*100+0.5), 693);
    BOOST_CHECK_EQUAL(int(ts.testEvalThreshold(2)*100+0.5), 173);
}

BOOST_AUTO_TEST_CASE(test_UpdateWeights)
{
    TestSoinn ts;
    ts.initializeNetwork();
    
    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 6;
    coords[1] = 6;
    coords[2] = 7;

    wv::WeightVectorContEuclidean wv(coords);
    ts.testFindWinners(&wv);
    
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(1);
     
    ts.testUpdateWeights(&wv);
    
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(0, 0), 1);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(0, 1), 2);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(0, 2), 3);
    
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(1, 0) * 1000 + 0.5), 3015);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(1, 1) * 100  + 0.5), 401);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(1, 2) * 100  + 0.5), 501);
    
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(2, 0), 4);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(2, 1), 3);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(2, 2), 6);
    
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 0) * 100 + 0.5), 675);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 1) * 100 + 0.5), 750);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 2) * 100 + 0.5), 850);
}

BOOST_AUTO_TEST_CASE(test_ProcessNewPoint)
{
    TestSoinn ts;
    ts.initializeNetwork();
    
    //first point. It has to be a new point
    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 6;
    coords[1] = 6;
    coords[2] = 7;

    wv::WeightVectorContEuclidean wv(coords);
    ts.testProcessNewPoint(&wv);
    
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(4, 0), 6);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(4, 1), 6);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(4, 2), 7);

    //--------------------------Second point---------------
    coords[0] = 6;
    coords[1] = 7;
    coords[2] = 8;
    
    wv::WeightVectorContEuclidean wv1(coords);
    ts.testProcessNewPoint(&wv1);

    BOOST_CHECK_EQUAL(ts.getWinner(), 4);
    BOOST_CHECK_EQUAL(ts.getSecWinner(), 3);
    
    BOOST_CHECK_EQUAL((int)(ts.getNeuron(ts.getWinner()).error() + 0.5), 2);
    
    std::vector<uint32_t> neighbours = ts.getNeuron(3).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 2);
    
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 1);
    BOOST_CHECK_EQUAL(neighbours.at(1), 4);
    
    BOOST_CHECK_EQUAL(ts.getNeuron(ts.getWinner()).localSignals(), 2);
    
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(1, 0), 3);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(1, 1), 4);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(1, 2), 5);
    
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 0) * 100 + 0.5), 699);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 1) * 100 + 0.5), 799);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 2) * 100 + 0.5), 899);
    
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(4, 0) * 100 + 0.5), 600);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(4, 1) * 100 + 0.5), 650);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(4, 2) * 100 + 0.5), 750);
    
    //----------------------------Third point-------------------
    coords[0] = 3;
    coords[1] = 7;
    coords[2] = 7;
    
    wv::WeightVectorContEuclidean wv2(coords);
    ts.testProcessNewPoint(&wv2);

    //---------------------------Forth point---------------------
    coords[0] = 5;
    coords[1] = 7;
    coords[2] = 7;
    
    wv::WeightVectorContEuclidean wv3(coords);
    ts.testProcessNewPoint(&wv3);
    
    //std::cout << ts.testEvalThreshold(1) << std::endl;
    //std::cout << ts.testEvalThreshold(4) << std::endl;
    
    BOOST_CHECK_EQUAL(ts.getWinner(), 4);
    BOOST_CHECK_EQUAL(ts.getSecWinner(), 5);
    
    BOOST_CHECK_EQUAL((int)(ts.getNeuron(ts.getWinner()).error()*10 + 0.5), 35);
    
    neighbours = ts.getNeuron(ts.getWinner()).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 1);
    
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 5);
    
    BOOST_CHECK_EQUAL(ts.getNeuron(ts.getWinner()).localSignals(), 3);
    
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(1, 0), 3);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(1, 1), 4);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(1, 2), 5);
    
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 0) * 100 + 0.5), 697);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 1) * 100 + 0.5), 798);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(3, 2) * 100 + 0.5), 897);
    
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(4, 0) * 100 + 0.5), 567);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(4, 1) * 100 + 0.5), 667);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(4, 2) * 100 + 0.5), 733);    

    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(5, 0) * 100 + 0.5), 302);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(5, 1) * 100 + 0.5), 700);
    BOOST_CHECK_EQUAL(int(ts.getNeuronCoord(5, 2) * 100 + 0.5), 700);    
}

BOOST_AUTO_TEST_SUITE_END()
