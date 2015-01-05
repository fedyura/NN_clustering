#include <boost/test/unit_test.hpp>
#include <weight_vector/WeightVectorContEuclidean.hpp>
#include <adapt_learn_rate/AdaptLearnRateKohonenSchema.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(WeightVector)

BOOST_AUTO_TEST_CASE(test_CalcDistance)
{
    cont::StaticArray<double> arr1(3);
    arr1[0] = 1;
    arr1[1] = 2;
    arr1[2] = 3;
    
    cont::StaticArray<double> arr2(3);
    arr2[0] = -1;
    arr2[1] = 4;
    arr2[2] = 2;

    wv::WeightVectorContEuclidean wv1(arr1);
    wv::WeightVectorContEuclidean wv2(arr2);
    wv::AbstractWeightVector* awv = &wv2;
    
    BOOST_CHECK_EQUAL(wv1.calcDistance(awv), 3);
    
    cont::StaticArray<double> arr3(4);
    arr2[0] = -1;
    arr2[1] = 4;
    arr2[2] = 2;
    arr3[3] = 4;
    wv::WeightVectorContEuclidean wv3(arr3);
    BOOST_CHECK_THROW(wv1.calcDistance(&wv3), std::runtime_error);
    BOOST_CHECK_THROW(wv3.calcDistance(awv), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(test_updateWeightVector)
{
    cont::StaticArray<double> arr1(3);
    arr1[0] = 1;
    arr1[1] = 2;
    arr1[2] = 3;
    
    cont::StaticArray<double> arr2(3);
    arr2[0] = -1;
    arr2[1] = 4;
    arr2[2] = 2;

    wv::WeightVectorContEuclidean wv1(arr1);
    wv::WeightVectorContEuclidean wv2(arr2);
    wv::AbstractWeightVector* awv2 = &wv2;
    wv::AbstractWeightVector* awv1 = &wv1;
    alr::KohonenParameters kp(10, 5, 1, 10);
    alr::AdaptLearnRateKohonenSchema alrks1(5, kp); //0.335
    alr::AdaptLearnRateKohonenSchema alrks2(5, kp); //0.606
    
    BOOST_REQUIRE_NO_THROW(awv1->updateWeightVector(awv2, &alrks1, 4));
    BOOST_CHECK_EQUAL(int(wv1.getConcreteCoord(0)*1000), 328);
    BOOST_CHECK_EQUAL(int(wv1.getConcreteCoord(1)*1000), 2671);
    BOOST_CHECK_EQUAL(int(wv1.getConcreteCoord(2)*1000), 2664);
    
    BOOST_REQUIRE_NO_THROW(awv1->updateWeightVector(awv2, &alrks2, 0));
    BOOST_CHECK_EQUAL(int(wv1.getConcreteCoord(0)*1000), -477);
    BOOST_CHECK_EQUAL(int(wv1.getConcreteCoord(1)*1000), 3477);
    BOOST_CHECK_EQUAL(int(wv1.getConcreteCoord(2)*1000), 2261);
}

BOOST_AUTO_TEST_CASE(test_getMiddlePoints)
{
    cont::StaticArray<double> arr1(3);
    arr1[0] = 1;
    arr1[1] = 2;
    arr1[2] = 3;
    
    cont::StaticArray<double> arr2(3);
    arr2[0] = -1;
    arr2[1] = 4;
    arr2[2] = 2;

    wv::WeightVectorContEuclidean wv1(arr1);
    wv::WeightVectorContEuclidean wv2(arr2);
    wv::AbstractWeightVector* awv2 = &wv2;
    wv::AbstractWeightVector* awv1 = &wv1;
    
    wv::AbstractWeightVector* aw_res = awv1->getMiddlePoints(awv2);
    BOOST_CHECK_EQUAL(aw_res->getConcreteCoord(0), 0);
    BOOST_CHECK_EQUAL(aw_res->getConcreteCoord(1), 3);
    BOOST_CHECK_EQUAL(aw_res->getConcreteCoord(2), 2.5);
    delete aw_res;
}
BOOST_AUTO_TEST_SUITE_END()
