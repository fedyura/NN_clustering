#ifndef __NEURON_NEURAL_GAS_NEURON_HPP__
#define __NEURON_NEURAL_GAS_NEURON_HPP__

#include <neuron/AbstractNeuron.hpp>
#include <unordered_map>

namespace neuron
{
    class NeuralGasNeuron: public AbstractNeuron
    {
    public:
        NeuralGasNeuron(wv::AbstractWeightVector* wv, double error = 0.0)
            : AbstractNeuron(wv)
            , m_Error(error)
        { }
            
        NeuralGasNeuron()
            : AbstractNeuron(NULL)
            , m_Error(0)
        { }
        
        void updateError()
        {
            m_Error += m_CurPointDist*m_CurPointDist;
        }
        
        //increment age of edges for this neuron and return vector of neighbours
        std::vector<uint32_t> incrementEdgesAge()
        {
            std::vector<uint32_t> neighbours;
            for (auto& s: m_Neighbours)
            {
                s.second++;
                neighbours.push_back(s.first);
            }
            return neighbours;
        }

        //increment age of concrete edge
        //throw std::out_of_range if this edge isn't found
        void incrementConcreteEdgeAge(uint32_t number)
        {
            m_Neighbours.at(number)++;
        }

        void updateEdge(int number)
        {
            //null age of existing edge or create new edge with null age
            m_Neighbours[number] = 0;
        }
        
        void deleteOldEdges(uint32_t age_max)
        {
            std::unordered_map<uint32_t, uint32_t>::iterator it = m_Neighbours.begin();
            while (it != m_Neighbours.end())
            {
                if (it->second > age_max)
                    it = m_Neighbours.erase(it);
                else 
                    it++;    
            }
        }
        
        std::vector<uint32_t> getNeighbours() const
        {
            std::vector<uint32_t> neighbours;
            for (auto& s: m_Neighbours)
                neighbours.push_back(s.first);
            return neighbours;
        }

        void replaceNeighbour(uint32_t number_old, uint32_t number_new)
        {
            std::unordered_map<uint32_t, uint32_t>::iterator it = m_Neighbours.find(number_old);
            if (it == m_Neighbours.end()) 
                throw std::runtime_error("Error in replaceNeighbour function. Old edge doesn't exist");
            
            m_Neighbours.emplace(number_new, 0);  //it->second (if set old value)
            m_Neighbours.erase(it);
        }

        void deleteConcreteNeighbour(uint32_t number)
        {
            m_Neighbours.erase(number);
        }
        
        double error() const
        {
            return m_Error;
        }

        void setError(double error)
        {
            m_Error = error;
        }

        void changeError(double koeff)
        {
            m_Error = koeff * m_Error;
        }

        uint32_t getNumNeighbours()
        {
            return m_Neighbours.size();
        }
            
    private:
        double m_Error; //error of neuron
        std::unordered_map<uint32_t, uint32_t> m_Neighbours; //Neigbour number => edge age
    };
}
#endif //__NEURON_NEURAL_GAS_NEURON
