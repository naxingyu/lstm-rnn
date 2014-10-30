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

#include "TrainableLayer.hpp"
#include "../helpers/JsonClasses.hpp"
#include "../Configuration.hpp"

#include <stdexcept>

#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>


namespace layers {

    template <typename TDevice>
    typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::_weightUpdates()
    {
        return m_weightUpdates;
    }

    template <typename TDevice>
    TrainableLayer<TDevice>::TrainableLayer(const helpers::JsonValue &layerChild, const helpers::JsonValue &weightsSection, 
                                            int inputWeightsPerBlock, int internalWeightsPerBlock, Layer<TDevice> &precedingLayer)
        : Layer<TDevice>           (layerChild, precedingLayer.parallelSequences(), precedingLayer.maxSeqLength())
        , m_precedingLayer         (precedingLayer)
        , m_inputWeightsPerBlock   (inputWeightsPerBlock)
        , m_internalWeightsPerBlock(internalWeightsPerBlock)
        , m_bias                   (layerChild->HasMember("bias") ? static_cast<real_t>((*layerChild)["bias"].GetDouble()) : 0)
        , m_learningRate           (layerChild->HasMember("learningRate") ? static_cast<real_t>((*layerChild)["learningRate"].GetDouble()) : -1)
    {
        //std::cout << "Creating layer " << this->name() << std::endl;
        // check if the bias value exists
        if (!layerChild->HasMember("bias"))
            throw std::runtime_error(std::string("Missing value 'bias' in layer '") + this->name() + "'");

        // extract the weights if they are given in the network file
        Cpu::real_vector weights;

        if (weightsSection.isValid() && weightsSection->HasMember(this->name().c_str())) {
            if (!weightsSection->HasMember(this->name().c_str()))
                throw std::runtime_error(std::string("Missing weights section for layer '") + this->name() + "'");
            const rapidjson::Value &weightsChild = (*weightsSection)[this->name().c_str()];
            if (!weightsChild.IsObject())
                throw std::runtime_error(std::string("Weights section for layer '") + this->name() + "' is not an object");

            if (!weightsChild.HasMember("input") || !weightsChild["input"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + this->name() + "/input'");
            if (!weightsChild.HasMember("bias") || !weightsChild["bias"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + this->name() + "/bias'");
            if (!weightsChild.HasMember("internal") || !weightsChild["internal"].IsArray())
                throw std::runtime_error(std::string("Missing array 'weights/") + this->name() + "/internal'");
        
            const rapidjson::Value &inputWeightsChild    = weightsChild["input"];
            const rapidjson::Value &biasWeightsChild     = weightsChild["bias"];
            const rapidjson::Value &internalWeightsChild = weightsChild["internal"];

            if (inputWeightsChild.Size() != this->size() * inputWeightsPerBlock * m_precedingLayer.size())
                throw std::runtime_error(std::string("Invalid number of input weights for layer '") + this->name() + "'");
            if (biasWeightsChild.Size() != this->size() * inputWeightsPerBlock)
                throw std::runtime_error(std::string("Invalid number of bias weights for layer '") + this->name() + "'");
            if (internalWeightsChild.Size() != this->size() * internalWeightsPerBlock)
                throw std::runtime_error(std::string("Invalid number of internal weights for layer '") + this->name() + "'");

            weights.reserve(inputWeightsChild.Size() + biasWeightsChild.Size() + internalWeightsChild.Size());

            for (rapidjson::Value::ConstValueIterator it = inputWeightsChild.Begin(); it != inputWeightsChild.End(); ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));
            for (rapidjson::Value::ConstValueIterator it = biasWeightsChild.Begin(); it != biasWeightsChild.End(); ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));
            for (rapidjson::Value::ConstValueIterator it = internalWeightsChild.Begin(); it != internalWeightsChild.End(); ++it)
                weights.push_back(static_cast<real_t>(it->GetDouble()));
        }
        // create random weights if no weights are given in the network file
        else {
            weights.resize(this->size() * (inputWeightsPerBlock * (m_precedingLayer.size() + 1) + internalWeightsPerBlock));

            const Configuration &config = Configuration::instance();

            static boost::mt19937 *gen = NULL;
            if (!gen) {
                gen = new boost::mt19937;
                gen->seed(config.randomSeed());
            }
            
            if (config.weightsDistributionType() == Configuration::DISTRIBUTION_UNIFORM) {
                real_t range = config.weightsDistributionUniformMax() - config.weightsDistributionUniformMin();
                boost::random::uniform_real_distribution<real_t> dist(0, range);
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = dist(*gen) + config.weightsDistributionUniformMin();
            }
            else {
                boost::random::normal_distribution<real_t> dist(config.weightsDistributionNormalMean(), config.weightsDistributionNormalSigma());
                for (size_t i = 0; i < weights.size(); ++i)
                    weights[i] = dist(*gen);
            }
        }

        m_weights       = weights;
        m_weightUpdates = weights;

        // resize the output errors vector
        //m_outputErrors = Cpu::real_vector(this->_outputs().size(), (real_t)0);
    }

    template <typename TDevice>
    TrainableLayer<TDevice>::~TrainableLayer()
    {
    }

    template <typename TDevice>
    Layer<TDevice>& TrainableLayer<TDevice>::precedingLayer()
    {
        return m_precedingLayer;
    }

    template <typename TDevice>
    const Layer<TDevice>& TrainableLayer<TDevice>::precedingLayer() const
    {
        return m_precedingLayer;
    }

    template <typename TDevice>
    real_t TrainableLayer<TDevice>::bias() const
    {
        return m_bias;
    }

    template <typename TDevice>
    real_t TrainableLayer<TDevice>::learningRate() const
    {
        return m_learningRate;
    }

/*    template <typename TDevice>
    typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::outputErrors()
    {
        return m_outputErrors;
    }*/

    template <typename TDevice>
    typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::weights()
    {
        return m_weights;
    }

    template <typename TDevice>
    const typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::weights() const
    {
        return m_weights;
    }

    template <typename TDevice>
    const typename TrainableLayer<TDevice>::real_vector& TrainableLayer<TDevice>::weightUpdates() const
    {
        return m_weightUpdates;
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::injectWeightNoise(real_t sigma) 
    {
        // generate vector of weight noise on the host
        // note: RNG is sequential, so we can't parallelize ...
        static boost::mt19937 *gen = NULL;
        if (!gen) {
            gen = new boost::mt19937;
            gen->seed(Configuration::instance().randomSeed());
        }
        boost::normal_distribution<real_t> dist(0.0f, sigma);
        Cpu::real_vector weightNoise(weights().size());
        for (int i = 0; i < weightNoise.size(); ++i) {
            weightNoise[i] = dist(*gen);
        }

        // copy weight noise to device
        real_vector weightNoiseD(weights().size());
        thrust::copy(weightNoise.begin(), weightNoise.end(), weightNoiseD.begin());

        // add weight noise to device vector of weights
        thrust::transform(weights().begin(), weights().end(), weightNoiseD.begin(), weights().begin(), thrust::plus<real_t>());
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::exportWeights(const helpers::JsonValue &weightsObject, const helpers::JsonAllocator &allocator) const
    {
        if (!weightsObject->IsObject())
            throw std::runtime_error("The JSON value is not an object");

        // do nothing if we don't have any weights
        if (m_weights.empty())
            return;

        // create and fill the weight arrays
        rapidjson::Value inputWeightsArray(rapidjson::kArrayType);
        int inputWeightsCount = this->size() * m_inputWeightsPerBlock * m_precedingLayer.size();
        inputWeightsArray.Reserve(inputWeightsCount, allocator);
        for (int i = 0; i < inputWeightsCount; ++i)
            inputWeightsArray.PushBack(m_weights[i], allocator);

        rapidjson::Value biasWeightsArray(rapidjson::kArrayType);
        int biasWeightsCount = this->size() * m_inputWeightsPerBlock;
        biasWeightsArray.Reserve(biasWeightsCount, allocator);
        for (int i = 0; i < biasWeightsCount; ++i)
            biasWeightsArray.PushBack(m_weights[inputWeightsCount + i], allocator);

        rapidjson::Value internalWeightsArray(rapidjson::kArrayType);
        int internalWeightsCount = this->size() * m_internalWeightsPerBlock;
        internalWeightsArray.Reserve(internalWeightsCount, allocator);
        for (int i = 0; i < internalWeightsCount; ++i)
            internalWeightsArray.PushBack(m_weights[inputWeightsCount + biasWeightsCount + i], allocator);

        // create and fill the weights subsection
        rapidjson::Value weightsSection(rapidjson::kObjectType);
        weightsSection.AddMember("input",    inputWeightsArray,    allocator);
        weightsSection.AddMember("bias",     biasWeightsArray,     allocator);
        weightsSection.AddMember("internal", internalWeightsArray, allocator);

        // add the weights section tot he weights object
        weightsObject->AddMember(this->name().c_str(), weightsSection, allocator);
    }

    template <typename TDevice>
    void TrainableLayer<TDevice>::exportLayer(const helpers::JsonValue &layersArray, const helpers::JsonAllocator &allocator) const
    {
        Layer<TDevice>::exportLayer(layersArray, allocator);
        (*layersArray)[layersArray->Size() - 1].AddMember("bias", m_bias, allocator);
    }


    // explicit template instantiations
    template class TrainableLayer<Cpu>;
    template class TrainableLayer<Gpu>;

} // namespace layers
