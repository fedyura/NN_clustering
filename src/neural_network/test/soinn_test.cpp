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
    const double TestC      = 0.8;
    const uint32_t TestLambda = 2;
}

class TestSoinn: public Soinn
{
  public:
    TestSoinn(double alpha1 = TestAlpha1, NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN)
      : Soinn(TestNumDimensions, alpha1, TestAlpha2, TestAlpha3, TestBetta, TestGamma, TestAgeMax, TestLambda, TestC, nnit, nt)
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

    void testInsertNode()
    {
        insertNode();
    }

    void testInsertConcreteEdge(int neur1, int neur2)
    {
        InsertConcreteEdge(neur1, neur2);
    }

    double testCalcAvgLocalSignals()
    {
        return calcAvgLocalSignals();
    }

    void testDeleteNeuron(uint32_t number)
    {
        deleteNeuron(number);
    }

    void testDeleteNodes()
    {
        deleteNodes();
    }

    double testCalcInnerClusterDistance()
    {
        return calcInnerClusterDistance();
    }

    void testInsertConcreteNeuron(const wv::Point* p)
    {
        InsertConcreteNeuron(p);
    }

    void testCalcBetweenClustersDistanceVector(const std::vector<std::vector<uint32_t>>& conn_comp, std::vector<double>& dist) const
    {
        calcBetweenClustersDistanceVector(conn_comp, dist);
    }

    double testCalcThresholdSecondLayer(const std::vector<std::vector<uint32_t>>& clusters) const
    {
        return calcThresholdSecondLayer(clusters);
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

    BOOST_CHECK_EQUAL((int)(ts.testCalcInnerClusterDistance()*1000 + 0.5), 5196);

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

    BOOST_CHECK_EQUAL((int)(ts.testCalcInnerClusterDistance()*1000+0.5), 4234);
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
    
    BOOST_CHECK_EQUAL((int)(ts.testCalcInnerClusterDistance()*1000+0.5), 4345);
}

BOOST_AUTO_TEST_CASE(test_InsertNode)
{
    //Insertion is successfull
    TestSoinn ts;
    ts.initializeNetwork();
    ts.testInsertConcreteEdge(0, 3);
    
    ts.setError(0, 4);
    ts.setError(1, 2);
    ts.setError(2, 3);
    ts.setError(3, 5);

    ts.incrementLocalSignals(2);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);

    ts.testInsertNode();
        
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(4, 0), 4);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(4, 1), 5);
    BOOST_CHECK_EQUAL(ts.getNeuronCoord(4, 2), 6);

    BOOST_CHECK_EQUAL(ts.getNeuron(0).error(), 2.4);
    BOOST_CHECK_EQUAL(ts.getNeuron(3).error(), 3);
    BOOST_CHECK_EQUAL(ts.getNeuron(4).error(), 1.8);

    BOOST_CHECK_EQUAL(ts.getNeuron(0).localSignals(), 0.8);
    BOOST_CHECK_EQUAL(ts.getNeuron(3).localSignals(), 3.2);
    BOOST_CHECK_EQUAL(ts.getNeuron(4).localSignals(), 2.5);

    BOOST_CHECK_EQUAL(ts.getNeuron(0).errorRadius(), 4);
    BOOST_CHECK_EQUAL((int)(ts.getNeuron(3).errorRadius()*100 + 0.5), 125);
    BOOST_CHECK_EQUAL((int)(ts.getNeuron(4).errorRadius()*1000 + 0.5), 3675);

    //Insertion isn't successfull
    TestSoinn ts1(0.8);
    ts1.initializeNetwork();
    ts1.testInsertConcreteEdge(0, 3);
    
    ts1.setError(0, 4);
    ts1.setError(1, 2);
    ts1.setError(2, 3);
    ts1.setError(3, 5);
    
    ts1.incrementLocalSignals(0);
    ts1.incrementLocalSignals(0);
    ts1.incrementLocalSignals(0);
    ts1.incrementLocalSignals(2);
    ts1.incrementLocalSignals(3);
    ts1.incrementLocalSignals(3);
    ts1.incrementLocalSignals(3);

    ts1.testInsertNode();
        
    BOOST_CHECK_EQUAL(ts1.getNeuron(0).error(), 4);
    BOOST_CHECK_EQUAL(ts1.getNeuron(3).error(), 5);

    BOOST_CHECK_EQUAL(ts1.getNeuron(0).localSignals(), 4);
    BOOST_CHECK_EQUAL(ts1.getNeuron(3).localSignals(), 4);

    BOOST_CHECK_EQUAL(ts1.getNeuron(0).errorRadius(), 1);
    BOOST_CHECK_EQUAL((int)(ts1.getNeuron(3).errorRadius()*100 + 0.5), 125);
}

BOOST_AUTO_TEST_CASE(test_deleteNeurons)
{
    TestSoinn ts;
    ts.initializeNetwork();
    ts.testInsertConcreteEdge(0, 3);
    
    ts.incrementLocalSignals(2);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);
    
    std::vector<uint32_t> neighbours = ts.getNeuron(0).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 2);
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 1);
    BOOST_CHECK_EQUAL(neighbours.at(1), 3);
    
    BOOST_CHECK_EQUAL(ts.testCalcAvgLocalSignals(), 2);
    BOOST_CHECK_EQUAL(ts.numEmptyNeurons(), 0);
    
    ts.testDeleteNeuron(1);
    ts.testDeleteNeuron(2);

    neighbours = ts.getNeuron(0).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 1);
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 3);
    
    BOOST_CHECK_EQUAL(ts.testCalcAvgLocalSignals(), 2.5);
    BOOST_CHECK_EQUAL(ts.numEmptyNeurons(), 2);
}

BOOST_AUTO_TEST_CASE(test_deleteNodes)
{
    TestSoinn ts;
    ts.initializeNetwork();

    ts.incrementLocalSignals(2);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);
    ts.incrementLocalSignals(3);
    
    ts.testDeleteNodes();
    
    BOOST_CHECK_EQUAL(ts.getNeuron(0).is_deleted(), true);
    BOOST_CHECK_EQUAL(ts.getNeuron(1).is_deleted(), false);
    BOOST_CHECK_EQUAL(ts.getNeuron(2).is_deleted(), true);
    BOOST_CHECK_EQUAL(ts.getNeuron(3).is_deleted(), false);
    
    std::vector<uint32_t> neighbours = ts.getNeuron(1).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 1);
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 3);

    neighbours = ts.getNeuron(3).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 1);
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 1);

    BOOST_CHECK_EQUAL(ts.testCalcAvgLocalSignals(), 2.5);
    BOOST_CHECK_EQUAL(ts.numEmptyNeurons(), 2);
}

BOOST_AUTO_TEST_CASE(test_FindConnectedComponents)
{
    TestSoinn ts;
    ts.initializeNetwork();

    cont::StaticArray<double> coords(TestNumDimensions);
    coords[0] = 10;
    coords[1] = 11;
    coords[2] = 12;
    wv::WeightVectorContEuclidean wv(coords);
    ts.testInsertConcreteNeuron(&wv);

    coords[0] = 11;
    coords[1] = 12;
    coords[2] = 13;
    wv::WeightVectorContEuclidean wv1(coords);
    ts.testInsertConcreteNeuron(&wv1);
    
    coords[0] = 12;
    coords[1] = 13;
    coords[2] = 14;
    wv::WeightVectorContEuclidean wv2(coords);
    ts.testInsertConcreteNeuron(&wv2);

    coords[0] = 12.5;
    coords[1] = 13.5;
    coords[2] = 14.5;
    wv::WeightVectorContEuclidean wv3(coords);
    ts.testInsertConcreteNeuron(&wv3);

    coords[0] = 14;
    coords[1] = 15;
    coords[2] = 16;
    wv::WeightVectorContEuclidean wv4(coords);
    ts.testInsertConcreteNeuron(&wv4);

    coords[0] = 15;
    coords[1] = 16;
    coords[2] = 17;
    wv::WeightVectorContEuclidean wv5(coords);
    ts.testInsertConcreteNeuron(&wv5);

    coords[0] = 16;
    coords[1] = 17;
    coords[2] = 18;
    wv::WeightVectorContEuclidean wv6(coords);
    ts.testInsertConcreteNeuron(&wv6);

    ts.testInsertConcreteEdge(0, 3);
    ts.testInsertConcreteEdge(0, 2);
    ts.testInsertConcreteEdge(3, 4);
    ts.testInsertConcreteEdge(4, 5);
    ts.testInsertConcreteEdge(5, 6);

    ts.testInsertConcreteEdge(7, 8);
    ts.testInsertConcreteEdge(8, 9);
    ts.testInsertConcreteEdge(7, 9);

    std::vector<std::vector<uint32_t>> conn_comp;
    ts.findConnectedComponents(conn_comp);

    BOOST_REQUIRE_EQUAL(conn_comp.size(), 3);
    BOOST_REQUIRE_EQUAL(conn_comp[0].size(), 7);
    std::sort(conn_comp[0].begin(), conn_comp[0].end());
    BOOST_CHECK_EQUAL(conn_comp[0][0], 0);
    BOOST_CHECK_EQUAL(conn_comp[0][1], 1);
    BOOST_CHECK_EQUAL(conn_comp[0][2], 2);
    BOOST_CHECK_EQUAL(conn_comp[0][3], 3);
    BOOST_CHECK_EQUAL(conn_comp[0][4], 4);
    BOOST_CHECK_EQUAL(conn_comp[0][5], 5);
    BOOST_CHECK_EQUAL(conn_comp[0][6], 6);

    BOOST_REQUIRE_EQUAL(conn_comp[1].size(), 3);
    std::sort(conn_comp[1].begin(), conn_comp[1].end());
    BOOST_CHECK_EQUAL(conn_comp[1][0], 7);
    BOOST_CHECK_EQUAL(conn_comp[1][1], 8);
    BOOST_CHECK_EQUAL(conn_comp[1][2], 9);
    
    BOOST_REQUIRE_EQUAL(conn_comp[2].size(), 1);
    BOOST_CHECK_EQUAL(conn_comp[2][0], 10);

    std::vector<double> dist;

    ts.testCalcBetweenClustersDistanceVector(conn_comp, dist);
    BOOST_REQUIRE_EQUAL(dist.size(), 3);
    BOOST_CHECK_EQUAL((int)(dist[0] * 1000 + 0.5), 866); 
    BOOST_CHECK_EQUAL((int)(dist[1] * 1000 + 0.5), 1732); 
    BOOST_CHECK_EQUAL((int)(dist[2] * 1000 + 0.5), 6928); 

    BOOST_CHECK_EQUAL((int)(ts.testCalcInnerClusterDistance() * 1000 + 0.5), 4246);  
    BOOST_CHECK_EQUAL((int)(ts.testCalcThresholdSecondLayer(conn_comp) * 1000 + 0.5), 6928);
}

BOOST_AUTO_TEST_SUITE_END()
