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

#include "PostOutputLayer.hpp"

#include <boost/lexical_cast.hpp>
#include <stdexcept>


namespace layers {

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_targets()
    {
        return this->outputs();
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_actualOutputs()
    {
        return m_precedingLayer.outputs();
    }

    template <typename TDevice>
    typename PostOutputLayer<TDevice>::real_vector& PostOutputLayer<TDevice>::_outputErrors()
    {
        return m_precedingLayer.outputErrors();
    }

    template <typename TDevice>
    PostOutputLayer<TDevice>::PostOutputLayer(
        const helpers::JsonValue &layerChild, 
        Layer<TDevice> &precedingLayer,
        int requiredSize,
        bool createOutputs)
        : Layer<TDevice>  (layerChild, precedingLayer.parallelSequences(), precedingLayer.maxSeqLength(), createOutputs)
        , m_precedingLayer(precedingLayer)
    {
        if (this->size() != requiredSize)
            throw std::runtime_error("Size mismatch: " + boost::lexical_cast<std::string>(this->size()) + " vs. " + boost::lexical_cast<std::string>(requiredSize));
    }

    template <typename TDevice>
    PostOutputLayer<TDevice>::~PostOutputLayer()
    {
    }

    template <typename TDevice>
    void PostOutputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        if (fraction.outputPatternSize() != this->size()) {
            throw std::runtime_error(std::string("Output layer size of ") + boost::lexical_cast<std::string>(this->size())
            + " != data target pattern size of " + boost::lexical_cast<std::string>(fraction.outputPatternSize()));
        }

        Layer<TDevice>::loadSequences(fraction);

        if (!this->_outputs().empty())
        	thrust::copy(fraction.outputs().begin(), fraction.outputs().end(), this->_outputs().begin());
    }


    // explicit template instantiations
    template class PostOutputLayer<Cpu>;
    template class PostOutputLayer<Gpu>;

} // namespace layers
