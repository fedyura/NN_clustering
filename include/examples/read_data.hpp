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
    std::vector<std::string> split(const std::string& s, char delim)
    {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) 
            elems.push_back(item);
        
        return elems;
    }

    bool readDataSet(const std::string& filename, const uint32_t num_dimensions, std::vector<std::shared_ptr<wv::Point>>& points, std::vector<std::string>& answers)
    {
        std::vector<std::vector<double>> coords;
        std::vector<std::pair<double, double>> min_max_values;
        min_max_values.resize(num_dimensions);
        for (uint32_t i = 0; i < num_dimensions; i++)
        {
            min_max_values[i].first = 0; //max value
            min_max_values[i].second = std::numeric_limits<double>::max(); //min value
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
            if (items.size() != num_dimensions + 1)
            {
                std::cerr << "Error! Wrong line format" << std::endl;
                return false;
            }
            
            std::vector<double> arr;
            arr.resize(num_dimensions);
            for (uint32_t i = 0; i < num_dimensions; i++)
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
            answers.push_back(items[num_dimensions]);
        }
        fi.close();
        //normalize values
        
        cont::StaticArray<double> arr(num_dimensions);
        for (const auto& p: coords)
        {
            for (uint32_t i = 0; i < num_dimensions; i++)
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
