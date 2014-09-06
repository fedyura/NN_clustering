#ifndef __EXAMPLES_READ_DATA_HPP__
#define __EXAMPLES_READ_DATA_HPP__

#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <weight_vector/WeightVectorContEuclidean.hpp>

namespace ex
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

#endif //__EXAMPLES_READ_DATA_HPP__
