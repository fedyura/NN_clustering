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
    const uint32_t NumDimensionsForestDataset = 54;
}

int main (int argc, char* argv[])
{
    //FormMCLInputFile("mcl_input");
    
    //std::vector<std::shared_ptr<wv::Point>> points;
    //std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    //for (uint32_t i = 1; i < 8; i++)
    // {
        //uint32_t i = 1;
        std::vector<std::shared_ptr<wv::Point>> points;
        std::vector<std::string> answers;

        log_netw->info("Hello world");
        //std::string suffix = std::to_string(i);
        if (!ex::readDataSet("test_all1", NumDimensionsForestDataset, points, answers))
        {
            std::cerr << "Error! DataSet can not be read" << std::endl;
        }
            
        std::string output_filename = "points"; //+ suffix;
        //25, 50, 1, 0.1
        nn::ESoinn ns(NumDimensionsForestDataset, 360 /*age_max*/, 2160 /*lambda*/, 0.5 /*C1*/, 0.01 /*C2*/); 
            
        std::vector<std::vector<uint32_t>> conn_comp;
        ns.trainNetwork(points, conn_comp, answers, 1);

        ns.printNetworkClustersFile(output_filename, conn_comp);
    //}
    
    /*
    for (uint32_t i = 0; i < conn_comp.size(); i++)
        for (uint32_t j = 0; j < conn_comp[i].size(); j++)
            neuron_clusters.emplace(conn_comp[i][j], i);
        
    //define clusters
    uint32_t i = 0;
    for (const auto p: points)
    {
        log_netw->info((boost::format("%d ") % ns.findPointCluster(p.get(), neuron_clusters)).str(), true);
        i++;
        if (i % 60 == 0) log_netw->info("\n", true);
    }
    */
    
    //ns.trainNetworkNoiseReduction(points, answers, 20);  
    //ns.printNetworkNodesFile(output_filename); 
    
    log_netw->info("\n", true);
    log_netw->info("The program is succesfully ended");
    return 0;
}
