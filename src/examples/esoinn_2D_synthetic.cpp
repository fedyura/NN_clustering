#include <boost/format.hpp>
#include <examples/read_data.hpp>
#include <cmath>
#include <quality_measures/quality_measures.hpp>
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
    
    if (!ex::readDataSet("../data/s4", NumDimensionsSynthetic, points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
        
    //0x3x14
    std::string output_filename = "points";
    //good result
    //0x2x4
    
    //nn::ESoinn ns(NumDimensionsSynthetic, 100 /*age_max*/, 50 /*lambda*/, 0.7 /*C1*/, 0.1 /*C2*/); //Круто для s2 работает
    
    
    //nn::ESoinn ns(NumDimensionsSynthetic, 50 /*age_max*/, 25 /*lambda*/, 0.1 /*C1*/, 0.05 /*C2*/); //Compound dataset mcl 1.2        
    
    
    
    
    
    
    //nn::ESoinn ns(NumDimensionsSynthetic, 25 /*age_max*/, 26 /*lambda*/, 0.1 /*C1*/, 0.05 /*C2*/); //Compound dataset mcl 1.2        
    
    
    //nn::ESoinn ns(NumDimensionsSynthetic, 100 /*age_max*/, 50 /*lambda*/, 0.9 /*C1*/, 0.9 /*C2*/); //S2 dataset mcl 1.2        10,20 итераций
    nn::ESoinn ns(NumDimensionsSynthetic, 15 /*age_max*/, 30 /*lambda*/, 0.9 /*C1*/, 0.9 /*C2*/); //S2 dataset mcl 1.2        10,20 итераций
    


    //----------------------------------------------------------------------------
    //nn::ESoinn ns(NumDimensionsSynthetic, 30 /*age_max*/, 15 /*lambda*/, 0.1 /*C1*/, 0.05 /*C2*/); //Pathbased dataset mcl 1.2    20,30 итераций    
    //nn::ESoinn ns(NumDimensionsSynthetic, 23 /*age_max*/, 25 /*lambda*/, 0.1 /*C1*/, 0.05 /*C2*/); //Compound dataset mcl 1.2     10,50 итераций   
    
    //nn::ESoinn ns(NumDimensionsSynthetic, 100 /*age_max*/, 50 /*lambda*/, 0.72 /*C1*/, 0.9 /*C2*/); //S2 dataset mcl 1.2        10,20 итераций
    
    //nn::ESoinn ns(NumDimensionsSynthetic, 15 /*age_max*/, 30 /*lambda*/, 0.9 /*C1*/, 0.9 /*C2*/); //S4 dataset mcl 1.2        10,20 итераций
    //----------------------------------------------
    



    //nn::ESoinn ns(NumDimensionsSynthetic, 50 /*age_max*/, 60 /*lambda*/, 0.7 /*C1*/, 0.1 /*C2*/); //Jain dataset mcl 1.2
    

    std::vector<std::vector<uint32_t>> conn_comp;
    ns.trainNetwork(points, conn_comp, answers, 20, 10);   
    
    std::cout << "Number of clusters " << conn_comp.size() << std::endl;
    
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
    std::vector<uint32_t> find_clusters;
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
        uint32_t cluster_id = ns.findPointCluster(p.get(), neuron_clusters);
        ofc << cluster_id << std::endl;      
        find_clusters.push_back(cluster_id);
    }

    of.close();
    ofc.close();
        
    qm::QualityMeasures qlm(answers, find_clusters);
    std::cout << "QUALITY MEASURES" << std::endl;
    std::cout << "Purity = " << qlm.evalPurity() << std::endl;
    std::cout << "Normalized mutual information = " << qlm.evalNormalizedMutualInformation() << std::endl;
    qlm.findStatisticalQualityMeasures();
    std::cout << "Rand index = " << qlm.randIndex() << std::endl;
    std::cout << "precision = " << qlm.precision() << std::endl;
    std::cout << "recall = " << qlm.recall() << std::endl;
    std::cout << "Fscore(1) = " << qlm.Fscore(1) << std::endl;
    
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
