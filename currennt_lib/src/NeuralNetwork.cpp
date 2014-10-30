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

#include "NeuralNetwork.hpp"
#include "Configuration.hpp"
#include "LayerFactory.hpp"
#include "layers/InputLayer.hpp"
#include "layers/PostOutputLayer.hpp"
#include "helpers/JsonClasses.hpp"

#include <vector>
#include <stdexcept>
#include <cassert>

#include <boost/foreach.hpp>


template <typename TDevice>
NeuralNetwork<TDevice>::NeuralNetwork(const helpers::JsonDocument &jsonDoc, int parallelSequences, int maxSeqLength, 
                                      int inputSizeOverride = -1, int outputSizeOverride = -1)
{
    try {
        // check the layers and weight sections
        if (!jsonDoc->HasMember("layers"))
            throw std::runtime_error("Missing section 'layers'");
        rapidjson::Value &layersSection  = (*jsonDoc)["layers"];

        if (!layersSection.IsArray())
            throw std::runtime_error("Section 'layers' is not an array");

        helpers::JsonValue weightsSection;
        if (jsonDoc->HasMember("weights")) {
            if (!(*jsonDoc)["weights"].IsObject())
                throw std::runtime_error("Section 'weights' is not an object");

            weightsSection = helpers::JsonValue(&(*jsonDoc)["weights"]);
        }

        // extract the layers
        for (rapidjson::Value::ValueIterator layerChild = layersSection.Begin(); layerChild != layersSection.End(); ++layerChild) {
            // check the layer child type
            if (!layerChild->IsObject())
                throw std::runtime_error("A layer section in the 'layers' array is not an object");

            // extract the layer type and create the layer
            if (!layerChild->HasMember("type"))
                throw std::runtime_error("Missing value 'type' in layer description");

            std::string layerType = (*layerChild)["type"].GetString();

            // override input/output sizes
            if (inputSizeOverride > 0 && layerType == "input") {
              (*layerChild)["size"].SetInt(inputSizeOverride);
            }
/*  Does not work yet, need another way to identify a) postoutput layer (last!) and then the corresponging output layer and type!
            if (outputSizeOverride > 0 && (*layerChild)["name"].GetString() == "output") {
              (*layerChild)["size"].SetInt(outputSizeOverride);
            }
            if (outputSizeOverride > 0 && (*layerChild)["name"].GetString() == "postoutput") {
              (*layerChild)["size"].SetInt(outputSizeOverride);
            }
*/
            try {
            	layers::Layer<TDevice> *layer;

                if (m_layers.empty())
                	layer = LayerFactory<TDevice>::createLayer(layerType, &*layerChild, weightsSection, parallelSequences, maxSeqLength);
                else
                    layer = LayerFactory<TDevice>::createLayer(layerType, &*layerChild, weightsSection, parallelSequences, maxSeqLength, m_layers.back().get());

                m_layers.push_back(boost::shared_ptr<layers::Layer<TDevice> >(layer));
            }
            catch (const std::exception &e) {
                throw std::runtime_error(std::string("Could not create layer: ") + e.what());
            }
        }

        // check if we have at least one input, one output and one post output layer
        if (m_layers.size() < 3)
            throw std::runtime_error("Not enough layers defined");

        // check if only the first layer is an input layer
        if (!dynamic_cast<layers::InputLayer<TDevice>*>(m_layers.front().get()))
            throw std::runtime_error("The first layer is not an input layer");

        for (size_t i = 1; i < m_layers.size(); ++i) {
            if (dynamic_cast<layers::InputLayer<TDevice>*>(m_layers[i].get()))
                throw std::runtime_error("Multiple input layers defined");
        }

        // check if only the last layer is a post output layer
        if (!dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers.back().get()))
            throw std::runtime_error("The last layer is not a post output layer");

        for (size_t i = 0; i < m_layers.size()-1; ++i) {
            if (dynamic_cast<layers::PostOutputLayer<TDevice>*>(m_layers[i].get()))
                throw std::runtime_error("Multiple post output layers defined");
        }

        // check if two layers have the same name
        for (size_t i = 0; i < m_layers.size(); ++i) {
            for (size_t j = 0; j < m_layers.size(); ++j) {
                if (i != j && m_layers[i]->name() == m_layers[j]->name())
                    throw std::runtime_error(std::string("Different layers have the same name '") + m_layers[i]->name() + "'");
            }
        }
    }
    catch (const std::exception &e) {
        throw std::runtime_error(std::string("Invalid network file: ") + e.what());
    }
}

template <typename TDevice>
NeuralNetwork<TDevice>::~NeuralNetwork()
{
}

template <typename TDevice>
const std::vector<boost::shared_ptr<layers::Layer<TDevice> > >& NeuralNetwork<TDevice>::layers() const
{
    return m_layers;
}

template <typename TDevice>
layers::InputLayer<TDevice>& NeuralNetwork<TDevice>::inputLayer()
{
    return static_cast<layers::InputLayer<TDevice>&>(*m_layers.front());
}

template <typename TDevice>
layers::TrainableLayer<TDevice>& NeuralNetwork<TDevice>::outputLayer()
{
    return static_cast<layers::TrainableLayer<TDevice>&>(*m_layers[m_layers.size()-2]);
}

template <typename TDevice>
layers::PostOutputLayer<TDevice>& NeuralNetwork<TDevice>::postOutputLayer()
{
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers.back());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
        layer->loadSequences(fraction);
}

template <typename TDevice>
void NeuralNetwork<TDevice>::computeForwardPass()
{
    BOOST_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers)
        layer->computeForwardPass();
}

template <typename TDevice>
void NeuralNetwork<TDevice>::computeBackwardPass()
{
    BOOST_REVERSE_FOREACH (boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers) {
        layer->computeBackwardPass();
    //std::cout << "output errors " << layer->name() << std::endl;
    //thrust::copy(layer->outputErrors().begin(), layer->outputErrors().end(), std::ostream_iterator<real_t>(std::cout, ";"));
    //std::cout << std::endl;
    }
}

template <typename TDevice>
real_t NeuralNetwork<TDevice>::calculateError() const
{
    return static_cast<layers::PostOutputLayer<TDevice>&>(*m_layers.back()).calculateError();
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportLayers(const helpers::JsonDocument& jsonDoc) const
{
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the layers array
    rapidjson::Value layersArray(rapidjson::kArrayType);

    // create the layer objects
    for (size_t i = 0; i < m_layers.size(); ++i)
        m_layers[i]->exportLayer(&layersArray, &jsonDoc->GetAllocator());

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("layers"))
        jsonDoc->RemoveMember("layers");

    // add the section to the JSON document
    jsonDoc->AddMember("layers", layersArray, jsonDoc->GetAllocator());
}

template <typename TDevice>
void NeuralNetwork<TDevice>::exportWeights(const helpers::JsonDocument& jsonDoc) const
{
    if (!jsonDoc->IsObject())
        throw std::runtime_error("JSON document root must be an object");

    // create the weights object
    rapidjson::Value weightsObject(rapidjson::kObjectType);

    // create the weight objects
    BOOST_FOREACH (const boost::shared_ptr<layers::Layer<TDevice> > &layer, m_layers) {
    	layers::TrainableLayer<TDevice> *trainableLayer = dynamic_cast<layers::TrainableLayer<TDevice>*>(layer.get());
        if (trainableLayer)
            trainableLayer->exportWeights(&weightsObject, &jsonDoc->GetAllocator());
    }

    // if the section already exists, we delete it first
    if (jsonDoc->HasMember("weights"))
        jsonDoc->RemoveMember("weights");

    // add the section to the JSON document
    jsonDoc->AddMember("weights", weightsObject, jsonDoc->GetAllocator());
}

template <typename TDevice>
std::vector<std::vector<std::vector<real_t> > > NeuralNetwork<TDevice>::getOutputs()
{
    layers::TrainableLayer<TDevice> &ol = outputLayer();
    
    std::vector<std::vector<std::vector<real_t> > > outputs;
    for (int patIdx = 0; patIdx < (int)ol.patTypes().size(); ++patIdx) {
        switch (ol.patTypes()[patIdx]) {
        case PATTYPE_FIRST:
            outputs.resize(outputs.size() + 1);

        case PATTYPE_NORMAL:
        case PATTYPE_LAST: {{
            Cpu::real_vector pattern(ol.outputs().begin() + patIdx * ol.size(), ol.outputs().begin() + (patIdx+1) * ol.size());
            int psIdx = patIdx % ol.parallelSequences();
            outputs[psIdx].push_back(std::vector<real_t>(pattern.begin(), pattern.end()));
            break;
        }}

        default:
            break;
        }
    }

    return outputs;
}


// explicit template instantiations
template class NeuralNetwork<Cpu>;
template class NeuralNetwork<Gpu>;
