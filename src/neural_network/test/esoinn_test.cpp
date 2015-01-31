#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
#include <neural_network/ESoinn.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>

using namespace boost::unit_test;
using namespace nn;

namespace 
{
    const uint32_t TestNumDimensions = 3;
    
    const double TestC1      = 0.8;
    const double TestC2      = 0.8;
    
    const uint32_t TestLambda = 2;
}

class TestESoinn: public ESoinn
{
  public:
    TestESoinn(double C1 = TestC1, double C2 = TestC2, NetworkStopCriterion nnit = NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType nt = neuron::NeuronType::EUCLIDEAN)
      : ESoinn(TestNumDimensions, C1, C2, nnit, nt)
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

    double testMidDistNeighbours(uint32_t num_neuron)
    {
        return midDistNeighbours(num_neuron);
    }

    void testInsertConcreteEdge(int neur1, int neur2)
    {
        InsertConcreteEdge(neur1, neur2);
    }

    void testInsertNeuron(const wv::Point* p, uint32_t num_class = 0)
    {
        InsertConcreteNeuron(p, num_class);
    }

    void testUpdateEdgeWinSecWin()
    {
        updateEdgeWinSecWin();
    }

    bool testIsNodeAxis(uint32_t num_neuron)
    {
        return isNodeAxis(num_neuron);
    }

    uint32_t testFindAxisForNode(uint32_t number)
    {
        return findAxisForNode(number);
    }

    void testLabelClasses(std::map<uint32_t, uint32_t>& node_axis)
    {
        labelClasses(node_axis);
    }

    void testFindMergedClasses(const std::map<uint32_t, uint32_t>& node_axis, std::map<uint32_t, std::set<uint32_t>>& merged_classes) const
    {
        findMergedClasses(node_axis, merged_classes);
    }

    void testAxisMapping(const std::map<uint32_t, std::set<uint32_t>>& src, std::map<uint32_t, uint32_t>& mapping) const
    {
        findAxesMapping(src, mapping);
    }
};

BOOST_AUTO_TEST_SUITE(TestESoinnFunctions)

BOOST_AUTO_TEST_CASE(test_InitializeNN)
{
    TestESoinn tn;
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
}

BOOST_AUTO_TEST_CASE(test_MidDistNeighbours)
{
    TestESoinn tn;
    tn.initializeNetwork();

    BOOST_CHECK_EQUAL(int(tn.testMidDistNeighbours(1)*100 + 0.5), 520);
    BOOST_CHECK_EQUAL(int(tn.testMidDistNeighbours(0)*100 + 0.5), 346);
    BOOST_CHECK_EQUAL(int(tn.testMidDistNeighbours(2)*100 + 0.5), 0);

    tn.testInsertConcreteEdge(1,2);
    BOOST_CHECK_EQUAL(int(tn.testMidDistNeighbours(1)*100 + 0.5), 404);
}

BOOST_AUTO_TEST_CASE(test_UpdateEdgeWinSecWin)
{
    TestESoinn tn;
    tn.initializeNetwork();
    tn.setWinner(2);
    tn.setSecWinner(3);
    tn.testUpdateEdgeWinSecWin();
    
    std::vector<uint32_t> neighbours = tn.getNeuron(3).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 2);
    
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 1);
    BOOST_CHECK_EQUAL(neighbours.at(1), 2);

    tn.setSecWinner(0);
    tn.getNeuron(2).setCurClass(4);
    tn.getNeuron(0).setCurClass(4);
    tn.testUpdateEdgeWinSecWin();
    
    neighbours = tn.getNeuron(2).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 2);
    
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 0);
    BOOST_CHECK_EQUAL(neighbours.at(1), 3);

    tn.getNeuron(0).setCurClass(3);
    tn.testUpdateEdgeWinSecWin();
    
    neighbours = tn.getNeuron(2).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 1);
    
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 3);

    neighbours = tn.getNeuron(0).getNeighbours();
    BOOST_REQUIRE_EQUAL(neighbours.size(), 1);
    
    std::sort(neighbours.begin(), neighbours.end());
    BOOST_CHECK_EQUAL(neighbours.at(0), 1);
}

BOOST_AUTO_TEST_CASE(test_IsNodeAxis)
{
    TestESoinn tn;
    tn.initializeNetwork();

    cont::StaticArray<double> coords4(TestNumDimensions);
    coords4[0] = 7;
    coords4[1] = 8;
    coords4[2] = 9;
    wv::WeightVectorContEuclidean wv4(coords4);
    tn.testInsertNeuron(&wv4);
    
    cont::StaticArray<double> coords5(TestNumDimensions);
    coords5[0] = 1;
    coords5[1] = 3;
    coords5[2] = 4;
    wv::WeightVectorContEuclidean wv5(coords5);
    tn.testInsertNeuron(&wv5);

    cont::StaticArray<double> coords6(TestNumDimensions);
    coords6[0] = 9;
    coords6[1] = 3;
    coords6[2] = 2;
    wv::WeightVectorContEuclidean wv6(coords6);
    tn.testInsertNeuron(&wv6);

    cont::StaticArray<double> coords7(TestNumDimensions);
    coords7[0] = 2;
    coords7[1] = 1;
    coords7[2] = 2;
    wv::WeightVectorContEuclidean wv7(coords7);
    tn.testInsertNeuron(&wv7);

    tn.testInsertConcreteEdge(1, 4);
    tn.testInsertConcreteEdge(4, 5);
    tn.testInsertConcreteEdge(3, 5);
    tn.testInsertConcreteEdge(0, 7);
    tn.testInsertConcreteEdge(7, 6);

    tn.getNeuron(0).setDensity(6);
    tn.getNeuron(1).setDensity(3);
    tn.getNeuron(2).setDensity(2);
    tn.getNeuron(3).setDensity(4);
    tn.getNeuron(4).setDensity(2);
    tn.getNeuron(5).setDensity(7);
    tn.getNeuron(6).setDensity(1);
    tn.getNeuron(7).setDensity(2);
    
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(0), true);
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(1), false);
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(2), true);
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(3), false);
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(4), false);
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(5), true);
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(6), false);
    BOOST_CHECK_EQUAL(tn.testIsNodeAxis(7), false);

    BOOST_CHECK_EQUAL(tn.getNeuron(0).neighbourMaxDensity(), 0);
    BOOST_CHECK_EQUAL(tn.getNeuron(1).neighbourMaxDensity(), 0);
    BOOST_CHECK_EQUAL(tn.getNeuron(2).neighbourMaxDensity(), 2);
    BOOST_CHECK_EQUAL(tn.getNeuron(3).neighbourMaxDensity(), 5);
    BOOST_CHECK_EQUAL(tn.getNeuron(4).neighbourMaxDensity(), 5);
    BOOST_CHECK_EQUAL(tn.getNeuron(5).neighbourMaxDensity(), 5);
    BOOST_CHECK_EQUAL(tn.getNeuron(6).neighbourMaxDensity(), 7);
    BOOST_CHECK_EQUAL(tn.getNeuron(7).neighbourMaxDensity(), 0);

    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(0), 0);
    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(1), 0);
    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(2), 2);
    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(3), 5);
    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(4), 5);
    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(5), 5);
    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(6), 0);
    BOOST_CHECK_EQUAL(tn.testFindAxisForNode(7), 0);

    std::map<uint32_t, uint32_t> node_axis;
    tn.testLabelClasses(node_axis);
    BOOST_CHECK_EQUAL(node_axis.size(), 8);
    BOOST_CHECK_EQUAL(node_axis.at(0), 0);
    BOOST_CHECK_EQUAL(node_axis.at(1), 0);
    BOOST_CHECK_EQUAL(node_axis.at(2), 2);
    BOOST_CHECK_EQUAL(node_axis.at(3), 5);
    BOOST_CHECK_EQUAL(node_axis.at(4), 5);
    BOOST_CHECK_EQUAL(node_axis.at(5), 5);
    BOOST_CHECK_EQUAL(node_axis.at(6), 0);
    BOOST_CHECK_EQUAL(node_axis.at(7), 0);
    
    tn.testInsertConcreteEdge(2, 1);
    std::map<uint32_t, std::set<uint32_t>> merged_classes;
    tn.testFindMergedClasses(node_axis, merged_classes);
    
    BOOST_REQUIRE_EQUAL(merged_classes.size(), 3);
    BOOST_CHECK_EQUAL(merged_classes.at(0).size(), 2);  
    BOOST_CHECK_EQUAL(merged_classes.at(5).size(), 1);  
    BOOST_CHECK_EQUAL(merged_classes.at(2).size(), 1);
      
    BOOST_CHECK_EQUAL(merged_classes.at(0).count(2), 1);
    BOOST_CHECK_EQUAL(merged_classes.at(0).count(5), 1);
    BOOST_CHECK_EQUAL(merged_classes.at(5).count(0), 1);
    BOOST_CHECK_EQUAL(merged_classes.at(2).count(0), 1);
}

BOOST_AUTO_TEST_CASE(test_AxesMapping)
{
    TestESoinn tn;
    
    std::map<uint32_t, std::set<uint32_t>> graph;
    graph[0].emplace(2);
    graph[0].emplace(3);

    graph[2].emplace(0);
    
    graph[3].emplace(0);
    graph[3].emplace(4);

    graph[4].emplace(3);
    graph[4].emplace(5);

    graph[5].emplace(4);
    graph[5].emplace(6);

    graph[6].emplace(5);
    
    graph[7].emplace(8);
    graph[7].emplace(9);

    graph[8].emplace(7);
    graph[8].emplace(9);

    graph[9].emplace(8);
    graph[9].emplace(7);
    
    std::set<uint32_t> empty;
    graph[1] = empty;
    
    std::map<uint32_t, uint32_t> mapping;
    tn.testAxisMapping(graph, mapping);
    BOOST_CHECK_EQUAL(mapping.size(), 10);
    BOOST_CHECK_EQUAL(mapping.at(0), 0);
    BOOST_CHECK_EQUAL(mapping.at(1), 1);
    BOOST_CHECK_EQUAL(mapping.at(2), 0);
    BOOST_CHECK_EQUAL(mapping.at(3), 0);
    BOOST_CHECK_EQUAL(mapping.at(4), 0);
    BOOST_CHECK_EQUAL(mapping.at(5), 0);
    BOOST_CHECK_EQUAL(mapping.at(6), 0);
    BOOST_CHECK_EQUAL(mapping.at(7), 7);
    BOOST_CHECK_EQUAL(mapping.at(8), 7);
    BOOST_CHECK_EQUAL(mapping.at(9), 7);
}

BOOST_AUTO_TEST_SUITE_END()
