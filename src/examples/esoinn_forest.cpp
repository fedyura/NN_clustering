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
    
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    if (!ex::readDataSet("forest_dataset", NumDimensionsForestDataset, points, answers))
    {
        std::cerr << "Error! DataSet can not be read" << std::endl;
    }
        
    std::string output_filename = "points";
    //25, 50, 1, 0.1
    nn::ESoinn ns(NumDimensionsForestDataset, 630 /*age_max*/, 1260 /*lambda*/, 1.0 /*C1*/, 0 /*C2*/); 
        
    ns.trainNetworkNoiseReduction(points, answers, 20);  
    ns.printNetworkNodesFile(output_filename); 
    
    log_netw->info("\n", true);
    log_netw->info("The program is succesfully ended");
    return 0;
}
