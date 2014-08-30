#include <boost/test/unit_test.hpp>
#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <adapt_learn_rate/AdaptLearnRateNeuralGas.hpp>
#include <adapt_learn_rate/AdaptLearnRateKohonenSchema.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(AdaptLearnRate)

BOOST_AUTO_TEST_CASE(test_KohonenSchema)
{
    alr::KohonenParameters kp1(10, 5, 1, 10);
    alr::AdaptLearnRateKohonenSchema alrks1(5, kp1);
    BOOST_CHECK_EQUAL(int(alrks1.getLearnRate(4)*1000), 335); 

    alr::AdaptLearnRateKohonenSchema alrks2(5, kp1);
    BOOST_CHECK_EQUAL(int(alrks2.getLearnRate(0)*1000), 606); 
}

BOOST_AUTO_TEST_CASE(test_NeuronGasSchema)
{
    alr::AdaptLearnRateNeuralGas alrng1(5, 0.1, 0.01);
    BOOST_CHECK_EQUAL(int(alrng1.getLearnRate(0.5)*1000), 10);
    BOOST_CHECK_EQUAL(int(alrng1.getLearnRate(0)*1000), 100);
}

BOOST_AUTO_TEST_CASE(test_SoinnSchema)
{
    alr::AdaptLearnRateSoinn alrs1(5, 4);
    BOOST_CHECK_EQUAL(alrs1.getLearnRate(0), 0.25);

    alr::AdaptLearnRateSoinn alrs2(5, 5);
    BOOST_CHECK_EQUAL(alrs2.getLearnRate(0.1), 0.002);
}

BOOST_AUTO_TEST_SUITE_END()
