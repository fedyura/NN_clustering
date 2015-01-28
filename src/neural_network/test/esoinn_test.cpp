#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <algorithm>
#include <boost/test/unit_test.hpp>
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

BOOST_AUTO_TEST_SUITE_END()
