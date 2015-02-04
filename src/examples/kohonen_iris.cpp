#include <boost/format.hpp>
#include <cmath>
#include <examples/read_data.hpp>
#include <logger/logger.hpp>
#include <neural_network/KohonenNN.hpp>

namespace
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("KohonenNetwork");
    const uint32_t NumDimensionsIrisDataset = 4;
    const uint32_t NumClustersIrisDataset = 3;
}

int main (int argc, char* argv[])
{
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    if (!ex::readDataSet("iris_dataset", NumDimensionsIrisDataset, points, answers))
    {
        log_netw->error("readIrisDataSet function works incorrect");
    }
        
    //0x3x14
    alr::KohonenParameters kp(0.4, 1.5, 0.9, 1.3);
    nn::KohonenNN knn(NumClustersIrisDataset, NumDimensionsIrisDataset, kp, 0); 

    knn.trainNetwork(points, 0.000001);
    
    log_netw->info("Results:");
    for (const auto p: points)
    {
        log_netw->info((boost::format("%d") % knn.getCluster(p.get())).str(), true);
    }    
    
    log_netw->info("\n", true);
    return 0;
}
