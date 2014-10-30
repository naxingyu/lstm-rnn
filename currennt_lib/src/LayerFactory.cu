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

#include "LayerFactory.hpp"

#include "layers/InputLayer.hpp"
#include "layers/FeedForwardLayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/LstmLayer.hpp"
#include "layers/SsePostOutputLayer.hpp"
#include "layers/RmsePostOutputLayer.hpp"
#include "layers/CePostOutputLayer.hpp"
#include "layers/SseMaskPostOutputLayer.hpp"
#include "layers/WeightedSsePostOutputLayer.hpp"
#include "layers/BinaryClassificationLayer.hpp"
#include "layers/MulticlassClassificationLayer.hpp"
#include "activation_functions/Tanh.cuh"
#include "activation_functions/Logistic.cuh"
#include "activation_functions/Identity.cuh"

#include <stdexcept>


template <typename TDevice>
layers::Layer<TDevice>* LayerFactory<TDevice>::createLayer(
		const std::string &layerType, const helpers::JsonValue &layerChild,
        const helpers::JsonValue &weightsSection, int parallelSequences, 
        int maxSeqLength, layers::Layer<TDevice> *precedingLayer)
{
    using namespace layers;
    using namespace activation_functions;

    if (layerType == "input")
    	return new InputLayer<TDevice>(layerChild, parallelSequences, maxSeqLength);
    else if (layerType == "feedforward_tanh")
    	return new FeedForwardLayer<TDevice, Tanh>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "feedforward_logistic")
    	return new FeedForwardLayer<TDevice, Logistic>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "feedforward_identity")
    	return new FeedForwardLayer<TDevice, Identity>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "softmax")
    	return new SoftmaxLayer<TDevice, Identity>(layerChild, weightsSection, *precedingLayer);
    else if (layerType == "lstm")
    	return new LstmLayer<TDevice>(layerChild, weightsSection, *precedingLayer, false);
    else if (layerType == "blstm")
    	return new LstmLayer<TDevice>(layerChild, weightsSection, *precedingLayer, true);
    else if (layerType == "sse" || layerType == "weightedsse" || layerType == "rmse" || layerType == "ce" || layerType == "wf" || layerType == "binary_classification" || layerType == "multiclass_classification") {
        //layers::TrainableLayer<TDevice>* precedingTrainableLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(precedingLayer);
        //if (!precedingTrainableLayer)
    	//    throw std::runtime_error("Cannot add post output layer after a non trainable layer");

        if (layerType == "sse")
    	    return new SsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "weightedsse")
    	    return new WeightedSsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "rmse")
            return new RmsePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "ce")
            return new CePostOutputLayer<TDevice>(layerChild, *precedingLayer);
        if (layerType == "sse_mask" || layerType == "wf") // wf provided for compat. with dev. version
    	    return new SseMaskPostOutputLayer<TDevice>(layerChild, *precedingLayer);
        else if (layerType == "binary_classification")
    	    return new BinaryClassificationLayer<TDevice>(layerChild, *precedingLayer);
        else // if (layerType == "multiclass_classification")
    	    return new MulticlassClassificationLayer<TDevice>(layerChild, *precedingLayer);
    }
    else
        throw std::runtime_error(std::string("Unknown layer type '") + layerType + "'");
}


// explicit template instantiations
template class LayerFactory<Cpu>;
template class LayerFactory<Gpu>;
