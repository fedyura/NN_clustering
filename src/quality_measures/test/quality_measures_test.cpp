#include <boost/test/unit_test.hpp>
#include <quality_measures/quality_measures.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(QualityMeasures)

BOOST_AUTO_TEST_CASE(test_StaticArrayWork)
{
    std::vector<std::string> classes;
    std::vector<uint32_t> clusters;
    
    classes.push_back("class1");
    classes.push_back("class2");
    classes.push_back("class3");
    classes.push_back("class3");
    classes.push_back("class3");
    classes.push_back("class2");
    classes.push_back("class3");
    classes.push_back("class1");
    classes.push_back("class3");
    classes.push_back("class3");

    clusters.push_back(1);
    clusters.push_back(3);
    clusters.push_back(4);
    clusters.push_back(5);
    clusters.push_back(2);
    clusters.push_back(4);
    clusters.push_back(2);
    clusters.push_back(4);
    clusters.push_back(4);
    clusters.push_back(4);
    
    qm::QualityMeasures qlm(classes, clusters); 
    BOOST_CHECK_EQUAL(qlm.evalPurity(), 0.8);
    //BOOST_CHECK_EQUAL((uint32_t)(qlm.evalEntropy(qlm.m_FreqClusters)*1000 + 0.5), 1961);
    //BOOST_CHECK_EQUAL((uint32_t)(qlm.evalEntropy(qlm.m_FreqClasses)*1000 + 0.5), 1371);
    //BOOST_CHECK_EQUAL((uint32_t)(qlm.evalMutualInformation()*1000 + 0.5), 685);
    
    BOOST_CHECK_EQUAL((uint32_t)(qlm.evalNormalizedMutualInformation()*1000 + 0.5), 411);

    qlm.findStatisticalQualityMeasures();
    BOOST_CHECK_EQUAL(qlm.TP(), 4);
    BOOST_CHECK_EQUAL(qlm.FP(), 7);
    BOOST_CHECK_EQUAL(qlm.TN(), 21);
    BOOST_CHECK_EQUAL(qlm.FN(), 13);
    
    BOOST_CHECK_EQUAL((uint32_t)(qlm.randIndex()*1000 + 0.5), 556);
    BOOST_CHECK_EQUAL((uint32_t)(qlm.precision()*1000 + 0.5), 364);
    BOOST_CHECK_EQUAL((uint32_t)(qlm.recall()*1000 + 0.5), 235);
    BOOST_CHECK_EQUAL((uint32_t)(qlm.Fscore(1)*1000 + 0.5), 286);
}

BOOST_AUTO_TEST_SUITE_END()
