#include <boost/format.hpp>
#include <examples/read_data.hpp>
#include <cmath>
#include <logger/logger.hpp>
#include <neural_network/NeuralGas.hpp>
#include <stdlib.h>

namespace 
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("NeuralGas");
    
    std::unordered_map<uint32_t, uint32_t> neuron_clusters; //neuron => cluster
    const uint32_t NumDimensionsSynthetic = 2;

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

    void FormMCLInputFile(const std::string& filename)
    {
        std::vector<std::shared_ptr<wv::Point>> points;
        std::vector<std::string> answers;

        if (!ex::readDataSet("aggregation", NumDimensionsSynthetic, points, answers))
        {
            std::cerr << "readIrisDataSet function works incorrect" << std::endl;
        }
        
        std::ofstream out(filename, std::ios::out);
        
        double threshold = 7, weight = 0;
        for (uint32_t i = 0; i < points.size(); i++)
        {
            wv::WeightVectorContEuclidean* sWeightVector = dynamic_cast<wv::WeightVectorContEuclidean*>(points[i].get()); 
            for (uint32_t j = i + 1; j < points.size(); j++)
            {
                weight = 1.0 / sWeightVector->calcDistance(points[j].get());
                if (weight < threshold)
                    continue;
                out << i << " " << j << " " << weight << std::endl;    
            }
        }

        out.close();
    }
}

int main (int argc, char* argv[])
{
    std::string output_filename = "mcl_input";
    FormMCLInputFile(output_filename);
    
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    
    log_netw->info("Hello world");
    
    if (!ex::readDataSet("aggregation", NumDimensionsSynthetic, points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
    
    /*    
    //0x3x14
    std::string output_filename = "points";
    //good result
    //0x2x4
    nn::NeuralGas ng(NumDimensionsIrisDataset, 0.2, 0.006, 0.5, 0.995, 100, 50); 
    ng.trainNetwork(points, 0.05);   
    ng.exportEdgesFile(output_filename);
    */
    
    //run mcl algorithm
    std::string mcl_path = "/home/yura/local/bin/mcl ";
    std::string output_mcl = "clusters";    
    std::string options = " -I 1.5 --abc -o ";
    std::string command = mcl_path + output_filename + options + output_mcl;
    system(command.c_str());
    
    //read answer from mcl
    readMCLAnswer(output_mcl);
    
    //define clusters
    /*
    for (const auto p: points)
    {
        log_netw->info((boost::format("%d") % ng.findPointCluster(p.get(), neuron_clusters)).str(), true);
    }
    */

    //print file for visualization
    std::ofstream of("visualize_file", std::ios::out);
    std::ofstream ofc("cluster_file", std::ios::out);
    for (uint32_t j = 0; j < points.size(); j++) 
    {
        auto p = points[j];
        for (uint32_t i = 0; i < p->getNumDimensions(); i++)
        {
            of << p->getConcreteCoord(i);
            ofc << p->getConcreteCoord(i) << " ";

            if (i != p->getNumDimensions() - 1)
                of << " ";
        }
        of << std::endl;
        try
        {
        ofc << neuron_clusters.at(j) << std::endl; //ns.findPointCluster(p.get(), neuron_clusters) << std::endl;      
        }
        catch(std::out_of_range& exc)
        {
            std::cout << "exception " << j << std::endl;
        }
    }
    of.close();
    ofc.close();
            
    log_netw->info("\n", true);
    log_netw->info("The program is succesfully ended");
    return 0;
}
