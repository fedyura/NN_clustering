#include <boost/format.hpp>
#include <examples/read_data.hpp>
#include <cmath>
#include <logger/logger.hpp>
#include <neural_network/ESoinn.hpp>
#include <stdlib.h>

namespace 
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("ESoinn", logger::LogLevels::DEBUG);
    
    std::unordered_map<uint32_t, uint32_t> neuron_clusters; //neuron => cluster
    const uint32_t NumDimensionsIrisDataset = 4;
}

int main (int argc, char* argv[])
{
    //FormMCLInputFile("mcl_input");
    
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    if (!ex::readDataSet("iris_dataset", NumDimensionsIrisDataset, points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
        
    std::string output_filename = "points";
    //25, 50, 1, 0.1
    nn::ESoinn ns(NumDimensionsIrisDataset, 50 /*age_max*/, 150 /*lambda*/, 1.0 /*C1*/, 0.1 /*C2*/); 
        
    std::vector<std::vector<uint32_t>> conn_comp;
    ns.trainNetwork(points, conn_comp, 20);  
    ns.dumpNetwork(); 
    
    for (uint32_t i = 0; i < conn_comp.size(); i++)
        for (uint32_t j = 0; j < conn_comp[i].size(); j++)
            neuron_clusters.emplace(conn_comp[i][j], i);
        
    //define clusters
    uint32_t i = 0;
    for (const auto p: points)
    {
        log_netw->info((boost::format("%d ") % ns.findPointCluster(p.get(), neuron_clusters)).str(), true);
        i++;
        if (i % 50 == 0) log_netw->info("\n", true);
    }
            
    log_netw->info("\n", true);
    log_netw->info("The program is succesfully ended");
    return 0;
}
