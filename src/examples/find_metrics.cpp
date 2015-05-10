#include <boost/format.hpp>
#include <examples/read_data.hpp>
#include <cmath>
#include <quality_measures/quality_measures.hpp>
#include <logger/logger.hpp>
#include <neural_network/ESoinn.hpp>
#include <stdlib.h>

namespace 
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("FindMetrics", logger::LogLevels::INFO);
    
    std::unordered_map<uint32_t, uint32_t> neuron_clusters; //neuron => cluster
    const uint32_t NumDimensionsSynthetic = 2;

    bool readDataClusters(const std::string& filename, const uint32_t num_dimensions, const uint32_t num_column, std::vector<uint32_t>& clusters, std::vector<std::string>& answers)
    {
        std::ifstream fi(filename);
        if (not fi)
        {
            std::cerr << "File " << filename << " doesn't exist" << std::endl;
            return false;
        }

        std::string line;
        //std::getline(fi, line); //skip first line
        uint32_t j = 0;
        while (std::getline(fi, line) and !line.empty())
        {
            std::vector<std::string> items = ex::split(line, ';');
            if (items.size() != num_dimensions + 1)
            {
                std::cerr << "Error! Wrong line format" << std::endl;
                return false;
            }
            
            if (j == 0)
            {
                std::cout << items[num_column] << std::endl;
                j++;
                continue;
            }

            for (uint32_t i = 0; i < num_dimensions; i++)
            {
                 std::istringstream iss(items[i]);
                 double number = 0;
                 if (!(iss >> number))
                 {
                     std::cerr << "Error! Point coord is not a number" << std::endl;
                     return false;
                 }
                 if (i == num_column)
                 {
                     clusters.push_back(number);
                     break;
                 }
            }
            answers.push_back(items[num_dimensions]);            
        }
        fi.close();
        return true;
    }
}

int main (int argc, char* argv[])
{
    //FormMCLInputFile("mcl_input");
    
    std::vector<uint32_t> clusters;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    if (!readDataClusters("/home/yura/SOINN/Results/copy/iris9", 18, 17, clusters, answers))
    {
        std::cerr << "readDataSet function works incorrect" << std::endl;
    }
        
    qm::QualityMeasures qlm(answers, clusters);
    std::cout << "QUALITY MEASURES" << std::endl;
    std::cout << "Purity = " << qlm.evalPurity() << std::endl;
    std::cout << "Normalized mutual information = " << qlm.evalNormalizedMutualInformation() << std::endl;
    qlm.findStatisticalQualityMeasures();
    std::cout << "Rand index = " << qlm.randIndex() << std::endl;
    std::cout << "precision = " << qlm.precision() << std::endl;
    std::cout << "recall = " << qlm.recall() << std::endl;
    std::cout << "Fscore(1) = " << qlm.Fscore(1) << std::endl;
        
    log_netw->info("\n", true);
    log_netw->info("The program is succesfully ended");
    return 0;
}
