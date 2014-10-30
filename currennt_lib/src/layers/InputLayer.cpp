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

#include "InputLayer.hpp"

#include <boost/lexical_cast.hpp>
#include <stdexcept>


namespace layers {

    template <typename TDevice>
    InputLayer<TDevice>::InputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength)
        : Layer<TDevice>(layerChild, parallelSequences, maxSeqLength)
    {
    }

    template <typename TDevice>
    InputLayer<TDevice>::~InputLayer()
    {
    }

    template <typename TDevice>
    const std::string& InputLayer<TDevice>::type() const
    {
        static const std::string s("input");
        return s;
    }

    template <typename TDevice>
    void InputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        if (fraction.inputPatternSize() != this->size()) {
            throw std::runtime_error(std::string("Input layer size of ") + boost::lexical_cast<std::string>(this->size())
            + " != data input pattern size of " + boost::lexical_cast<std::string>(fraction.inputPatternSize()));
        }

        Layer<TDevice>::loadSequences(fraction);

        thrust::copy(fraction.inputs().begin(), fraction.inputs().end(), this->_outputs().begin());
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeForwardPass()
    {
    }

    template <typename TDevice>
    void InputLayer<TDevice>::computeBackwardPass()
    {
    }


    // explicit template instantiations
    template class InputLayer<Cpu>;
    template class InputLayer<Gpu>;

} // namespace layers
