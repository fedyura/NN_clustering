#include <examples/read_data.hpp>
#include <cmath>
#include <neural_network/NeuralGas.hpp>
#include <stdlib.h>

namespace 
{
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

    void FormMCLInputFile(const std::string& filename)
    {
        std::vector<std::shared_ptr<wv::Point>> points;
        std::vector<std::string> answers;

        //std::cout << "Hello world" << std::endl;
    
        if (!ex::readIrisDataSet("iris_dataset", points, answers))
        {
            std::cerr << "readIrisDataSet function works incorrect" << std::endl;
        }
        
        std::ofstream out(filename, std::ios::out);
        
        double threshold = 2.5, weight = 0;
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
    //FormMCLInputFile("mcl_input");
    
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    std::cout << "Hello world" << std::endl;
    
    if (!ex::readIrisDataSet("iris_dataset", points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
        
    //0x3x14
    std::string output_filename = "points";
    //good result
    //0x2x4
    nn::NeuralGas ng(ex::NumDimensionsIrisDataSet, 0.2, 0.006, 0.5, 0.995, 100, 50); 
    ng.trainNetwork(points, 0.05);   
    ng.exportEdgesFile(output_filename);
    
    //run mcl algorithm
    std::string mcl_path = "/home/yura/local/bin/mcl ";
    std::string output_mcl = "clusters";    
    std::string options = " -I 2.0 --abc -o ";
    std::string command = mcl_path + output_filename + options + output_mcl;
    system(command.c_str());
    
    //read answer from mcl
    readMCLAnswer(output_mcl);
    
    //define clusters
    for (const auto p: points)
    {
        std::cout << ng.findPointCluster(p.get(), neuron_clusters);
    }
            
    std::cout << "The program is successfully ended" << std::endl;
    return 0;
    
}
