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

#ifndef LAYERS_POSTOUTPUTLAYER_HPP
#define LAYERS_POSTOUTPUTLAYER_HPP

#include "TrainableLayer.hpp"


namespace layers {

    /******************************************************************************************//**
     * This layer is used as the very last layer which store the target outputs and calculates the
     * error between the target outputs and the actual outputs
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class PostOutputLayer : public Layer<TDevice>
    {
        typedef typename TDevice::real_vector real_vector;

    private:
        Layer<TDevice> &m_precedingLayer;

    protected:
        real_vector& _targets();
        real_vector& _actualOutputs();
        real_vector& _outputErrors();

    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param precedingLayer The layer preceding this one
         * @param createOutputs  If false, then the outputs vector will be left empty
         */
        PostOutputLayer(
            const helpers::JsonValue &layerChild, 
            Layer<TDevice>  &precedingLayer,
            int requiredSize,
            bool                      createOutputs = true
            );

        /**
         * Destructs the Layer
         */
        virtual ~PostOutputLayer();

        /**
         * @see Layer::loadSequences()
         */
        virtual void loadSequences(const data_sets::DataSetFraction &fraction);

        /**
         * Computes the error with respect to the target outputs
         *
         * @return The error 
         */
        virtual real_t calculateError() =0;

    };

} // namespace layers


#endif // LAYERS_POSTOUTPUTLAYER_HPP
