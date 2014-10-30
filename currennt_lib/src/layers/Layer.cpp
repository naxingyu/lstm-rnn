/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifdef _MSC_VER
#   pragma warning (disable: 4244)
#endif

#include "Layer.hpp"
#include "../helpers/JsonClasses.hpp"

#include <stdexcept>


namespace layers {

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::_outputs()
    {
        return m_outputs;
    }

    template <typename TDevice>
    Layer<TDevice>::Layer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength, bool createOutputs)
        : m_name             (layerChild->HasMember("name") ? (*layerChild)["name"].GetString()  : "")
        , m_size             (layerChild->HasMember("size") ? (*layerChild)["size"].GetInt()     : 0)
        , m_parallelSequences(parallelSequences)
        , m_maxSeqLength     (maxSeqLength)
        , m_curMaxSeqLength  (0)
        , m_curMinSeqLength  (0)
        , m_curNumSeqs       (0)
    {
        // check if the name and size values exist
        if (!layerChild->HasMember("name"))
            throw std::runtime_error("Missing value 'name' in layer description");
        if (m_name.empty())
            throw std::runtime_error("Empty layer name in layer description");
        if (!layerChild->HasMember("size"))
            throw std::runtime_error(std::string("Missing value 'size' in layer '") + m_name + "'");

        // allocate space for the vectors
        if (createOutputs)
            m_outputs = Cpu::real_vector(m_parallelSequences * m_maxSeqLength * m_size);

        m_patTypes = Cpu::pattype_vector(m_parallelSequences * m_maxSeqLength);

        // resize the output errors vector
        m_outputErrors = Cpu::real_vector(this->_outputs().size(), (real_t)0);
    }

    template <typename TDevice>
    Layer<TDevice>::~Layer()
    {
    }

    template <typename TDevice>
    const std::string& Layer<TDevice>::name() const
    {
        return m_name;
    }

    template <typename TDevice>
    int Layer<TDevice>::size() const
    {
        return m_size;
    }

    template <typename TDevice>
    int Layer<TDevice>::parallelSequences() const
    {
        return m_parallelSequences;
    }

    template <typename TDevice>
    int Layer<TDevice>::maxSeqLength() const
    {
        return m_maxSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curMaxSeqLength() const
    {
        return m_curMaxSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curMinSeqLength() const
    {
        return m_curMinSeqLength;
    }

    template <typename TDevice>
    int Layer<TDevice>::curNumSeqs() const
    {
        return m_curNumSeqs;
    }

    template <typename TDevice>
    const typename Layer<TDevice>::pattype_vector& Layer<TDevice>::patTypes() const
    {
        return m_patTypes;
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::outputs()
    {
        return m_outputs;
    }

    template <typename TDevice>
    typename Layer<TDevice>::real_vector& Layer<TDevice>::outputErrors()
    {
        return m_outputErrors;
    }

    template <typename TDevice>
    void Layer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        m_curMaxSeqLength = fraction.maxSeqLength();
        m_curMinSeqLength = fraction.minSeqLength();
        m_curNumSeqs      = fraction.numSequences();
        m_patTypes        = fraction.patTypes();
    }
    
    template <typename TDevice>
    void Layer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, const helpers::JsonAllocator &allocator) const
    {
        if (!layersArray->IsArray())
            throw std::runtime_error("The JSON value is not an array");

        // create and fill the layer object
        rapidjson::Value layerObject(rapidjson::kObjectType);
        layerObject.AddMember("name", name().c_str(), allocator);
        layerObject.AddMember("type", type().c_str(), allocator);
        layerObject.AddMember("size", size(),         allocator);

        // add the layer object to the layers array
        layersArray->PushBack(layerObject, allocator);
    }


    // explicit template instantiations
    template class Layer<Cpu>;
    template class Layer<Gpu>;

} // namespace layers
