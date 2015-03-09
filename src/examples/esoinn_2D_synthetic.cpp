#include <boost/format.hpp>
#include <examples/read_data.hpp>
#include <cmath>
#include <logger/logger.hpp>
#include <neural_network/ESoinn.hpp>
#include <stdlib.h>

namespace 
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("Soinn", logger::LogLevels::INFO);
    
    std::unordered_map<uint32_t, uint32_t> neuron_clusters; //neuron => cluster
    const uint32_t NumDimensionsSynthetic = 2;
}

int main (int argc, char* argv[])
{
    //FormMCLInputFile("mcl_input");
    
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    if (!ex::readDataSet("s2.txt", NumDimensionsSynthetic, points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
        
    //0x3x14
    std::string output_filename = "points";
    //good result
    //0x2x4
    
    nn::ESoinn ns(NumDimensionsSynthetic, 50 /*age_max*/, 25 /*lambda*/, 0.7 /*C1*/, 0 /*C2*/); //Круто для s2 работает
    //nn::ESoinn ns(NumDimensionsSynthetic, 100 /*age_max*/, 50 /*lambda*/, 0.7 /*C1*/, 0.1 /*C2*/); //Jain dataset mcl 1.2
    //nn::ESoinn ns(NumDimensionsSynthetic, 100 /*age_max*/, 50 /*lambda*/, 0.8 /*C1*/, 0.1 /*C2*/); //Aggregation dataset


            
    std::vector<std::vector<uint32_t>> conn_comp;
    ns.trainNetwork(points, conn_comp, answers, 50);   
    
    for (uint32_t i = 0; i < conn_comp.size(); i++)
        for (uint32_t j = 0; j < conn_comp[i].size(); j++)
            neuron_clusters.emplace(conn_comp[i][j], i);
        
    /*
    ns.exportEdgesFile(output_filename);
    
    //run mcl algorithm
    std::string mcl_path = "/home/yura/local/bin/mcl ";
    std::string output_mcl = "clusters";    
    std::string options = " -I 2.0 --abc -o ";
    std::string command = mcl_path + output_filename + options + output_mcl;
    system(command.c_str());
    
    //read answer from mcl
    readMCLAnswer(output_mcl);
    */
        
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
        ofc << ns.findPointCluster(p.get(), neuron_clusters) << std::endl;      
    }
    of.close();
    ofc.close();
        
    //define clusters
    /*
    uint32_t i = 0;
    for (const auto p: points)
    {
        log_netw->info((boost::format("%d ") % ns.findPointCluster(p.get(), neuron_clusters)).str(), true);
        i++;
        if (i % 50 == 0) log_netw->info("\n", true);
    }
    */
            
    log_netw->info("\n", true);
    log_netw->info("The program is succesfully ended");
    return 0;
}
