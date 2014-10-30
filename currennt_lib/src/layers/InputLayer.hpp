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

#ifndef LAYERS_INPUTLAYER_HPP
#define LAYERS_INPUTLAYER_HPP

#include "Layer.hpp"


namespace layers {

    /******************************************************************************************//**
     * Represents the input layer of the neural network
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class InputLayer : public Layer<TDevice>
    {
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild        The layer section of the JSON configuration
         * @param parallelSequences The maximum number of sequences that shall be computed in parallel
         * @param maxSeqLength      The maximum length of a sequence
         */
        InputLayer(const helpers::JsonValue &layerChild, int parallelSequences, int maxSeqLength);

        /**
         * Destructs the Layer
         */
        virtual ~InputLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see Layer::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();
    };

} // namespace layers


#endif // LAYERS_INPUTLAYER_HPP
