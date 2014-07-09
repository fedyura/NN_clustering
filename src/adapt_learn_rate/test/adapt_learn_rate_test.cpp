#include <boost/test/unit_test.hpp>
#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <adapt_learn_rate/AdaptLearnRateMartines.hpp>
#include <adapt_learn_rate/AdaptLearnRateKohonenSchema.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(AdaptLearnRate)

BOOST_AUTO_TEST_CASE(test_KohonenSchema)
{
    alr::AdaptLearnRateKohonenSchema alrks1(5, 4, 10, 5, 1, 10);
    BOOST_CHECK_EQUAL(int(alrks1.getLearnRate()*1000), 335); 

    alr::AdaptLearnRateKohonenSchema alrks2(5, 0, 10, 5, 1, 10);
    BOOST_CHECK_EQUAL(int(alrks2.getLearnRate()*1000), 606); 
}

BOOST_AUTO_TEST_CASE(test_NeuronGasSchema)
{
    alr::AdaptLearnRateMartines alrm1(5, 0.5, 20, 0.1, 0.01);
    BOOST_CHECK_EQUAL(int(alrm1.getLearnRate()*1000), 56);

    alr::AdaptLearnRateMartines alrm2(0, 0.5, 20, 0.1, 0.001);
    BOOST_CHECK_EQUAL(alrm2.getLearnRate(), 0.1);
    
    alr::AdaptLearnRateMartines alrm3(20, 0.5, 20, 0.1, 0.001);
    BOOST_CHECK_EQUAL(alrm3.getLearnRate(), 0.001);
}

BOOST_AUTO_TEST_CASE(test_SoinnSchema)
{
    alr::AdaptLearnRateSoinn alrs1(5, 0, 4);
    BOOST_CHECK_EQUAL(alrs1.getLearnRate(), 0.25);

    alr::AdaptLearnRateSoinn alrs2(5, 0.1, 5);
    BOOST_CHECK_EQUAL(alrs2.getLearnRate(), 0.002);
}

BOOST_AUTO_TEST_SUITE_END()
