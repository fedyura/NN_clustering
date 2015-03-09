#include <boost/format.hpp>
#include <cmath>
#include <examples/read_data.hpp>
#include <logger/logger.hpp>
#include <neural_network/KohonenNN.hpp>

namespace
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("KohonenNetwork");
    const uint32_t NumDimensionsSynthetic = 2;
    const uint32_t NumClustersSynthetic = 7;
}

int main (int argc, char* argv[])
{
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    if (!ex::readDataSet("aggregation", NumDimensionsSynthetic, points, answers))
    {
        log_netw->error("readIrisDataSet function works incorrect");
    }
        
    //0x3x14
    alr::KohonenParameters kp(1.4, 2.2, 1.0, 4);
    nn::KohonenNN knn(NumClustersSynthetic, NumDimensionsSynthetic, kp, 0); 

    knn.trainNetwork(points, 0.000001);
    
    log_netw->info("Results:");
    for (const auto p: points)
    {
        //log_netw->info((boost::format("%d") % knn.getCluster(p.get())).str(), true);
    }    
    
    
    
    //print file for visualization
    std::ofstream of("visualize_file", std::ios::out);
    std::ofstream ofc("cluster_file", std::ios::out);
    for (const auto p: points)
    {
        for (uint32_t i = 0; i < p->getNumDimensions(); i++)
        {
            of << p->getConcreteCoord(i);
            ofc << p->getConcreteCoord(i) << " ";

            if (i != p->getNumDimensions() - 1)
                of << " ";
        }
        of << std::endl;
        ofc << knn.getCluster(p.get()) << std::endl;      
    }
    of.close();
    ofc.close();
    
    log_netw->info("\n", true);
    return 0;
}
