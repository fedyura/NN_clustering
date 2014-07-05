#include <boost/test/unit_test.hpp>
#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>
#include <adapt_learn_rate/AdaptLearnRateNeuronGas.hpp>
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

BOOST_AUTO_TEST_SUITE_END()
