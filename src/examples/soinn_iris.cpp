#include <boost/format.hpp>
#include <examples/read_data.hpp>
#include <cmath>
#include <logger/logger.hpp>
#include <neural_network/Soinn.hpp>
#include <stdlib.h>

namespace 
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("Soinn", logger::LogLevels::DEBUG);
    
    std::unordered_map<uint32_t, uint32_t> neuron_clusters; //neuron => cluster

    void readMCLAnswer(const std::string& filename)
    {
        std::ifstream fi(filename);
        if (not fi)
            throw std::runtime_error("File " + filename + " is not found");
        
        uint32_t linesread = 0;
        std::string line;
        while (std::getline(fi, line) and !line.empty())
        {
            std::vector<std::string> items = ex::split(line, '\t');
            for (const std::string& s: items)
            {
                uint32_t neuron_num = std::stoul(s);
                neuron_clusters.emplace(neuron_num, linesread);
            }
            linesread++;
        }
    }
}

int main (int argc, char* argv[])
{
    //FormMCLInputFile("mcl_input");
    
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    if (!ex::readIrisDataSet("iris_dataset", points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
        
    //0x3x14
    std::string output_filename = "points";
    //good result
    //0x2x4
    //nn::Soinn ns(ex::NumDimensionsIrisDataSet, 0.167, 0.25, 0.25, 0.667, 0.75, 100 /*age_max*/, 50 /*lambda*/, 0.5); 
    
    nn::Soinn ns(ex::NumDimensionsIrisDataSet, 0.167, 0.25, 0.25, 0.667, 0.75, 100 /*age_max*/, 50 /*lambda*/, 0.5); 
    ns.trainNetwork(points, 0.001);   
    ns.exportEdgesFile(output_filename);
    
    //run mcl algorithm
    std::string mcl_path = "/home/yura/local/bin/mcl ";
    std::string output_mcl = "clusters";    
    std::string options = " -I 2.0 --abc -o ";
    std::string command = mcl_path + output_filename + options + output_mcl;
    system(command.c_str());
    
    //read answer from mcl
    readMCLAnswer(output_mcl);
    
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
