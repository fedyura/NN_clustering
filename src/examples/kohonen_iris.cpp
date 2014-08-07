#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <neural_network/KohonenNN.hpp>
#include <memory>
#include <sstream>
#include <tuple>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace
{
    const uint32_t NumDimensionsIrisDataSet = 4;
    const uint32_t NumClustersIrisDataSet = 3;

    std::vector<std::string> split(const std::string& s, char delim)
    {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) 
            elems.push_back(item);
        
        return elems;
    }

    bool readIrisDataSet(const std::string& filename, std::vector<std::shared_ptr<wv::Point>>& points, std::vector<std::string>& answers)
    {
        std::vector<std::array<double, NumDimensionsIrisDataSet>> coords;
        std::array<std::pair<double, double>, NumDimensionsIrisDataSet> min_max_values; //min and max values by each coord
        for (uint32_t i = 0; i < NumDimensionsIrisDataSet; i++)
        {
            min_max_values[i].first = 0; //max value
            min_max_values[i].second = 1000; //min value
        }
                
        std::ifstream fi(filename);
        if (not fi)
        {
            std::cerr << "File " << filename << " doesn't exist" << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(fi, line) and !line.empty())
        {
            std::vector<std::string> items = split(line, ',');
            if (items.size() != 5)
            {
                std::cerr << "Error! Wrong line format" << std::endl;
                return false;
            }
            
            std::array<double, NumDimensionsIrisDataSet> arr;
            for (uint32_t i = 0; i < NumDimensionsIrisDataSet; i++)
            {
                 std::istringstream iss(items[i]);
                 double number = 0;
                 if (!(iss >> number))
                 {
                     std::cerr << "Error! Point coord is not a number" << std::endl;
                     return false;
                 }
                 arr[i] = number;
                 min_max_values[i].first = std::max(min_max_values[i].first, number);
                 min_max_values[i].second = std::min(min_max_values[i].second, number);
            }
            coords.push_back(arr);
            answers.push_back(items[4]);
        }
        fi.close();
        //normalize values
        
        cont::StaticArray<double> arr(NumDimensionsIrisDataSet);
        for (const auto& p: coords)
        {
            for (uint32_t i = 0; i < NumDimensionsIrisDataSet; i++)
            {
                arr[i] = (p[i] - min_max_values[i].second) / (min_max_values[i].first - min_max_values[i].second);
            }
            
            std::shared_ptr<wv::WeightVectorContEuclidean> swv(new wv::WeightVectorContEuclidean(arr));
            points.push_back(swv);
        }
        return true;
    }
}

int main (int argc, char* argv[])
{
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    std::cout << "Hello world" << std::endl;
    
    if (!readIrisDataSet("iris_dataset", points, answers))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
        
    //0x3x14
    alr::KohonenParameters kp(0.4, 1.5, 0.9, 1.3);
    nn::KohonenNN knn(NumClustersIrisDataSet, NumDimensionsIrisDataSet, kp, 0); 

    knn.trainNetwork(points, 0.000001);
    
    for (const auto p: points)
    {
        std::cout << knn.getCluster(p.get());
    }    
    
    std::cout << std::endl;
    return 0;
}
