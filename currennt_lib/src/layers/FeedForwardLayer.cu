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
#   pragma warning (disable: 4244) // thrust/iterator/iterator_adaptor.h(121): warning C4244: '+=' : conversion from '__int64' to 'int', possible loss of data
#endif

#include "FeedForwardLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Tanh.cuh"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Identity.cuh"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <typeinfo>


namespace internal {
namespace {

    template <typename TActFn>
    struct ComputeOutputFn
    {
        int    layerSize;
        real_t bias;

        const real_t *biasWeights;

        __host__ __device__ real_t operator() (real_t a, const int &outputIdx) const
        {
            // calculate indices
            int blockIdx = outputIdx % layerSize; 

            // add the bias
            a += bias * biasWeights[blockIdx];

            // apply the activation function
            real_t b = TActFn::fn(a);

            // store the activation
            return b;
        }
    };

    template <typename TActFn>
    struct ComputeDeltaFn
    {
        // since calculating the derivatives is very cheap for our activation functions, 
        // we simple calculate the deltas of all timesteps, including dummies
        
        __host__ __device__ void operator() (const thrust::tuple<real_t&, const real_t&> &t) const
        {
            real_t delta = TActFn::deriv(t.get<1>()) * t.get<0>();
            t.get<0>() = delta;
        }
    };

    struct ComputeBiasWeightUpdateFn
    {
        int    layerSize;
        int    patternsCount;
        real_t bias;

        const real_t *deltas;
        
        __host__ __device__ real_t operator() (const int &biasWeightIdx) const
        {
            const real_t *offDeltas = deltas + biasWeightIdx;

            real_t wu = 0;
            for (int i = 0; i < patternsCount; ++i) {
                wu += bias * *offDeltas;
                offDeltas += layerSize;
            }

            return wu;
        }
    };

} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice, typename TActFn>
    FeedForwardLayer<TDevice, TActFn>::FeedForwardLayer(
        const helpers::JsonValue &layerChild, 
        const helpers::JsonValue &weightsSection,
        Layer<TDevice> &precedingLayer)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 1, 0, precedingLayer)
    {
    }

    template <typename TDevice, typename TActFn>
    FeedForwardLayer<TDevice, TActFn>::~FeedForwardLayer()
    {
    }

    template <typename TDevice, typename TActFn>
    const std::string& FeedForwardLayer<TDevice, TActFn>::type() const
    {
        static std::string s;

        if (s.empty()) {
            if (typeid(TActFn) == typeid(activation_functions::Tanh))
                s = "feedforward_tanh";
            else if (typeid(TActFn) == typeid(activation_functions::Logistic))
                s = "feedforward_logistic";
            else if (typeid(TActFn) == typeid(activation_functions::Identity))
                s = "feedforward_identity";
            else
                throw std::runtime_error("Unsupported activation function");
        }
        
        return s;
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeForwardPass()
    {
        // collect outputs from preceding layer
        {{
            helpers::Matrix<TDevice> weightsMatrix  (&this->weights(),                  this->precedingLayer().size(), this->size());
            helpers::Matrix<TDevice> plOutputsMatrix(&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());
            helpers::Matrix<TDevice> outputsMatrix  (&this->_outputs(),                 this->size(),                  this->curMaxSeqLength() * this->parallelSequences());

            outputsMatrix.assignProduct(weightsMatrix, true, plOutputsMatrix, false);
        }}

        // calculate the outputs of the layer
        {{
            internal::ComputeOutputFn<TActFn> fn;
            fn.layerSize        = this->size();
            fn.bias             = this->bias();
            fn.biasWeights      = helpers::getRawPointer(this->weights()) + this->size() * this->precedingLayer().size();

            thrust::transform(
                this->_outputs().begin(),
                this->_outputs().begin() + this->curMaxSeqLength() * this->parallelSequences() * this->size(),
                thrust::counting_iterator<int>(0),
                this->_outputs().begin(),
                fn
                );
        }}
    }

    template <typename TDevice, typename TActFn>
    void FeedForwardLayer<TDevice, TActFn>::computeBackwardPass()
    {
        // compute deltas
        {{
            internal::ComputeDeltaFn<TActFn> fn;

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   this->outputs().begin())),
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, this->outputs().begin()+n)),
                fn
                );
        }}

        // back-propagate the error to the preceding layer
        {{
            TrainableLayer<TDevice> *pl = dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
            if (pl) {
                helpers::Matrix<TDevice> weightsMatrix (&this->weights(),      pl->size(),   this->size());
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(),   pl->size(),   this->curMaxSeqLength() * this->parallelSequences());
                helpers::Matrix<TDevice> deltasMatrix  (&this->outputErrors(), this->size(), this->curMaxSeqLength() * this->parallelSequences());

                plErrorsMatrix.assignProduct(weightsMatrix, false, deltasMatrix, false);
            }
        }}

        // compute the input weight updates
        {{
            helpers::Matrix<TDevice> weightUpdatesMatrix(&this->_weightUpdates(),           this->precedingLayer().size(), this->size());
            helpers::Matrix<TDevice> plOutputsMatrix    (&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());
            helpers::Matrix<TDevice> deltasMatrix       (&this->outputErrors(),             this->size(),                  this->curMaxSeqLength() * this->parallelSequences());

            weightUpdatesMatrix.assignProduct(plOutputsMatrix, false, deltasMatrix, true);
        }}

        // compute the bias weight updates
        {{
            internal::ComputeBiasWeightUpdateFn fn;
            fn.layerSize     = this->size();
            fn.patternsCount = this->curMaxSeqLength() * this->parallelSequences();
            fn.bias          = this->bias();
            fn.deltas        = helpers::getRawPointer(this->outputErrors());

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->size(),
                this->_weightUpdates().begin() + this->precedingLayer().size() * this->size(),
                fn
                );
        }}
    }


    // explicit template instantiations
    template class FeedForwardLayer<Cpu, activation_functions::Tanh>;
    template class FeedForwardLayer<Gpu, activation_functions::Tanh>;
    template class FeedForwardLayer<Cpu, activation_functions::Logistic>;
    template class FeedForwardLayer<Gpu, activation_functions::Logistic>;
    template class FeedForwardLayer<Cpu, activation_functions::Identity>;
    template class FeedForwardLayer<Gpu, activation_functions::Identity>;

} // namespace layers
