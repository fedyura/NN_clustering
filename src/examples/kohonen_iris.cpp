#include <examples/read_data.hpp>
#include <cmath>
#include <neural_network/KohonenNN.hpp>

int main (int argc, char* argv[])
{
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    std::cout << "Hello world" << std::endl;
    
    if (!ex::readIrisDataSet("iris_dataset", points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
        
    //0x3x14
    alr::KohonenParameters kp(0.4, 1.5, 0.9, 1.3);
    nn::KohonenNN knn(ex::NumClustersIrisDataSet, ex::NumDimensionsIrisDataSet, kp, 0); 

    knn.trainNetwork(points, 0.000001);
    
    for (const auto p: points)
    {
        std::cout << knn.getCluster(p.get());
    }    
    
    std::cout << std::endl;
    return 0;
}
