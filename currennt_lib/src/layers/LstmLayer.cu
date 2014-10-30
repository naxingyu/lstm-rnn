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

#include "LstmLayer.hpp"
#include "../helpers/limitedError.cuh"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/Matrix.hpp"
#include "../activation_functions/Logistic.cuh"
#include "../activation_functions/Tanh.cuh"

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>


namespace internal {
namespace {

    typedef activation_functions::Logistic gate_act_fn_t;
    typedef activation_functions::Tanh     cell_input_act_fn_t;
    typedef activation_functions::Tanh     cell_output_act_fn_t;

    struct ComputeBlockOutputFn
    {
        int    effLayerSize;
        int    prevOutputDistance;
        real_t bias;

        const char *patTypes;

        const real_t *niBiasWeights;
        const real_t *igBiasWeights;
        const real_t *fgBiasWeights;
        const real_t *ogBiasWeights;

        const real_t *igPeepWeights;
        const real_t *fgPeepWeights;
        const real_t *ogPeepWeights;

        real_t *cellStates;
        real_t *niActs;
        real_t *igActs;
        real_t *fgActs;
        real_t *ogActs;

        __host__ __device__ real_t operator() (const int &outputIdx, const thrust::tuple<bool, bool> &t) const
        {
            // unpack the tuple
            bool firstCall    = t.get<0>();
            bool checkPatType = t.get<1>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set the all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    if (prevOutputDistance > 0)
                        cellStates[outputIdx] = 0;
                    return 0;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

            // load the niag activations
            real_t niAct = niActs[outputIdx];
            real_t igAct = igActs[outputIdx];
            real_t fgAct = fgActs[outputIdx];
            real_t ogAct = ogActs[outputIdx];

            // add bias activations
            niAct += bias * niBiasWeights[blockIdx];
            igAct += bias * igBiasWeights[blockIdx];
            fgAct += bias * fgBiasWeights[blockIdx];
            ogAct += bias * ogBiasWeights[blockIdx];

            // add activation from peephole weights
            if (!firstCall) {
                real_t prevCellState = cellStates[outputIdx + prevOutputDistance];

                igAct += prevCellState * igPeepWeights[blockIdx];
                fgAct += prevCellState * fgPeepWeights[blockIdx];
            }

            // apply the activation functions
            niAct = cell_input_act_fn_t::fn(niAct);
            igAct = gate_act_fn_t      ::fn(igAct);
            fgAct = gate_act_fn_t      ::fn(fgAct);

            // store the niag activations
            niActs[outputIdx] = niAct;
            igActs[outputIdx] = igAct;
            fgActs[outputIdx] = fgAct;

            // calculate the cell state and store the result
            real_t cellState = niAct * igAct;

            if (!firstCall)
                cellState += cellStates[outputIdx + prevOutputDistance] * fgAct;

            cellStates[outputIdx] = cellState;

            // calculate the output gate activation and store the result
            ogAct += cellState * ogPeepWeights[blockIdx];
            ogAct = gate_act_fn_t::fn(ogAct);
            ogActs[outputIdx] = ogAct;

            // calculate the block output
            real_t output = cell_output_act_fn_t::fn(cellState) * ogAct;

            return output;
        }
    };

    struct ResortOutputsFn
    {
        int layerSize;
        int effLayerSize;

        const real_t *fwOutputs;
        const real_t *bwOutputs;

        __host__ __device__ real_t operator() (const int &outputIdx) const
        {
            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                return fwOutputs[offset];
            else
                return bwOutputs[offset - effLayerSize];
        }
    };

    struct ResortOutputErrorsFn
    {
        int layerSize;
        int effLayerSize;

        real_t *fwOutputErrors;
        real_t *bwOutputErrors;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int> &t) const
        {
            // unpack the tuple
            real_t outputErr = t.get<0>();
            int    outputIdx = t.get<1>();

            // calculate indices
            int patIdx = outputIdx / layerSize;
            int valIdx = outputIdx % layerSize;
            int offset = patIdx * effLayerSize + valIdx;

            // store the value
            if (valIdx < effLayerSize)
                fwOutputErrors[offset] = outputErr;
            else
                bwOutputErrors[offset - effLayerSize] = outputErr;
        }
    };

    struct ComputeBlockErrorsFn
    {
        int effLayerSize;
        int prevOutputDistance;

        const char *patTypes;

        const real_t *igPeepWeights;
        const real_t *fgPeepWeights;
        const real_t *ogPeepWeights;

        const real_t *cellStates;
        const real_t *niActs;
        const real_t *igActs;
        const real_t *fgActs;
        const real_t *ogActs;

        real_t *cellStateErrors;
        real_t *niDeltas;
        real_t *igDeltas;
        real_t *fgDeltas;
        real_t *ogDeltas;

        __host__ __device__ void operator() (const thrust::tuple<const real_t&, int, bool, bool, bool> &t) const
        {
            // unpack the tuple
            real_t outputErr    = t.get<0>();
            int    outputIdx    = t.get<1>();
            bool   firstCall    = t.get<2>();
            bool   lastCall     = t.get<3>();
            bool   checkPatType = t.get<4>();

            // check if we can skip the whole calculation because the pattern is a dummy
            // in that case, we set all values of that pattern to zero
            if (checkPatType) {
                int patIdx = outputIdx / effLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    niDeltas       [outputIdx] = 0;
                    igDeltas       [outputIdx] = 0;
                    fgDeltas       [outputIdx] = 0;
                    ogDeltas       [outputIdx] = 0;
                    cellStateErrors[outputIdx] = 0;
                    return;
                }
            }

            // calculate indices
            int blockIdx = outputIdx % effLayerSize;

            // load the niag activations, the cell state and the output error
            real_t niAct     = niActs      [outputIdx];
            real_t igAct     = igActs      [outputIdx];
            real_t ogAct     = ogActs      [outputIdx];
            real_t cellState = cellStates  [outputIdx];

            // calculate the output gate delta
            real_t ogDelta = gate_act_fn_t::deriv(ogAct) * cell_output_act_fn_t::fn(cellState) * outputErr;

            // calculate the cell state error
            real_t ogPeepWeight = ogPeepWeights[blockIdx];
            real_t cellStateErr = ogAct * cell_output_act_fn_t::deriv(cell_output_act_fn_t::fn(cellState)) * outputErr + ogPeepWeight * ogDelta;

            if (!firstCall) {
                real_t nextFgAct        = fgActs         [outputIdx - prevOutputDistance];
                real_t nextCellStateErr = cellStateErrors[outputIdx - prevOutputDistance];
                real_t nextIgDelta      = igDeltas       [outputIdx - prevOutputDistance];
                real_t nextFgDelta      = fgDeltas       [outputIdx - prevOutputDistance];
                
                real_t igPeepWeight = igPeepWeights[blockIdx];
                real_t fgPeepWeight = fgPeepWeights[blockIdx];

                cellStateErr += nextFgAct * nextCellStateErr + igPeepWeight * nextIgDelta + fgPeepWeight * nextFgDelta;
            }

            // calculate the net input delta
            real_t niDelta = igAct * cell_input_act_fn_t::deriv(niAct) * cellStateErr;

            // calculate the forget gate delta
            real_t fgDelta = 0;

            if (!lastCall) {
                real_t fgAct         = fgActs    [outputIdx];
                real_t prevCellState = cellStates[outputIdx + prevOutputDistance];

                fgDelta = gate_act_fn_t::deriv(fgAct) * prevCellState * cellStateErr;
            }

            // calculate the input gate delta
            real_t igDelta = gate_act_fn_t::deriv(igAct) * niAct * cellStateErr;

            // store the niag deltas and the cell state error
            niDeltas       [outputIdx] = helpers::limitedError(niDelta);
            igDeltas       [outputIdx] = helpers::limitedError(igDelta);
            fgDeltas       [outputIdx] = helpers::limitedError(fgDelta);
            ogDeltas       [outputIdx] = helpers::limitedError(ogDelta);
            cellStateErrors[outputIdx] = cellStateErr;
        }
    };

    struct ComputeWeightUpdateFn
    {
        int    layerSize;
        int    effLayerSize;
        int    precLayerSize;
        int    timestepDistance;
        int    parallelSequences;
        int    patternsCount;
        int    biasWeightsOffset;
        int    internalWeightsOffset;
        int    peepholeWeightsOffset;
        real_t bias;

        const real_t *plOutputs;
        const real_t *fwOutputs;   
        const real_t *bwOutputs;   
        const real_t *fwCellStates;
        const real_t *bwCellStates;
        const real_t *fwNiDeltas;  
        const real_t *bwNiDeltas;  
        const real_t *fwIgDeltas;  
        const real_t *bwIgDeltas;  
        const real_t *fwFgDeltas;  
        const real_t *bwFgDeltas;  
        const real_t *fwOgDeltas;  
        const real_t *bwOgDeltas;  

        __host__ __device__ real_t operator() (const int &weightIdx) const
        {
            // determine the weight type
            // 
            // weightType = 0bXXYY with XX = {input, bias, internal, peephole}
            //                     and  YY = {NI, IG, FG, OG}
            //
            // weightType = 0b0000 ( 0): NI input weight
            //              0b0001 ( 1): IG input weight
            //              0b0010 ( 2): FG input weight
            //              0b0011 ( 3): OG input weight
            //              0b0100 ( 4): NI bias weight
            //              0b0101 ( 5): IG bias weight
            //              0b0110 ( 6): FG bias weight
            //              0b0111 ( 7): OG bias weight
            //              0b1000 ( 8): NI internal weight
            //              0b1001 ( 9): IG internal weight
            //              0b1010 (10): FG internal weight
            //              0b1011 (11): OG internal weight
            //              0b1100 (12): not used
            //              0b1101 (13): IG peephole weight
            //              0b1110 (14): FG peephole weight
            //              0b1111 (15): OG peephole weight
            int inwc = layerSize * precLayerSize;
            int biwc = layerSize;
            int itwc = layerSize * effLayerSize;
            int pewc = layerSize;

            int weightType = (int)(weightIdx >= 0                     + 1 * inwc) +
                             (int)(weightIdx >= 0                     + 2 * inwc) +
                             (int)(weightIdx >= 0                     + 3 * inwc) +
                             (int)(weightIdx >= 0                     + 4 * inwc) + 
                             (int)(weightIdx >= biasWeightsOffset     + 1 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 2 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 3 * biwc) +
                             (int)(weightIdx >= biasWeightsOffset     + 4 * biwc) +
                             (int)(weightIdx >= internalWeightsOffset + 1 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 2 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 3 * itwc) +
                             (int)(weightIdx >= internalWeightsOffset + 4 * itwc) * 2 +
                             (int)(weightIdx >= peepholeWeightsOffset + 1 * pewc) +
                             (int)(weightIdx >= peepholeWeightsOffset + 2 * pewc);

            int weightTypeX = weightType & 0xC;
            int weightTypeY = weightType & 0x3;

            // calculate indices, offsets and increments 
            const real_t *offOutputs;
            int           tgtBlockIdx;
            int           offOutputsInc;
            bool          skipFirstPattern = false;
            bool          skipLastPattern  = false;
            bool          isBwStateWeight;

            switch (weightTypeX) {
            // input weight
            case 0x0: 
                {{
                    // calculate indices
                    int inputWeightIdx = weightIdx;
                    int plBlockIdx     = inputWeightIdx % precLayerSize;
                    int blockIdx       = (inputWeightIdx - weightTypeY * (biasWeightsOffset/4)) / precLayerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;
                    
                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &plOutputs[plBlockIdx];
                    offOutputsInc = precLayerSize;
                }}
                break;

            // bias weight
            case 0x4: 
                {{
                    // calculate indices
                    int biasWeightIdx = weightIdx - biasWeightsOffset;
                    int blockIdx      = biasWeightIdx - weightTypeY * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = NULL;
                    offOutputsInc = 0;
                }}
                break;

            // internal weight
            case 0x8: 
                {{
                    // calculate indices
                    int internalWeightIdx = weightIdx - internalWeightsOffset;
                    int srcBlockIdx       = internalWeightIdx % effLayerSize;
                    int blockIdx          = internalWeightIdx / effLayerSize - weightTypeY * layerSize;

                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = (isBwStateWeight ? &bwOutputs[srcBlockIdx] : &fwOutputs[srcBlockIdx]);
                    offOutputsInc = effLayerSize;

                    if (isBwStateWeight) {
                        offOutputs += timestepDistance;
                        skipLastPattern = true;
                    }
                    else {
                        offOutputs -= timestepDistance;
                        skipFirstPattern = true;
                    }
                }}
                break;

            // peephole weight
            default: 
                {{
                    // calculate indices
                    int peepholeWeightIdx = weightIdx - peepholeWeightsOffset;
                    int blockIdx          = peepholeWeightIdx - (weightTypeY-1) * layerSize;
                    
                    // check if we calculate backward state weights and adjust the block index
                    isBwStateWeight = (blockIdx >= effLayerSize);
                    if (isBwStateWeight)
                        blockIdx -= effLayerSize;

                    // select the appropriate cell states and adjust the block index
                    const real_t *cellStates = (isBwStateWeight ? bwCellStates : fwCellStates);
                    
                    // set the timeshift
                    int timeShift;
                    if (weightTypeY == 0x3) {
                        timeShift = 0;
                    }
                    else {
                        if (isBwStateWeight) {
                            timeShift       = timestepDistance;
                            skipLastPattern = true;
                        }
                        else {
                            timeShift        = -timestepDistance;
                            skipFirstPattern = true;
                        }
                    }

                    // set values for the loop below
                    tgtBlockIdx   = blockIdx;
                    offOutputs    = &cellStates[blockIdx + timeShift];
                    offOutputsInc = effLayerSize;
                }}
                break;
            }

            // determine the start of the delta values
            const real_t *niagDeltasLut[] = {
                fwNiDeltas,
                fwIgDeltas,
                fwFgDeltas,
                fwOgDeltas,
                bwNiDeltas,
                bwIgDeltas,
                bwFgDeltas,
                bwOgDeltas
            };

            // calculate the weight update over all patterns            
            const real_t *offDeltas = &niagDeltasLut[weightTypeY + (isBwStateWeight ? 4 : 0)][tgtBlockIdx];

            if (skipFirstPattern) {
                offOutputs += parallelSequences * offOutputsInc;
                offDeltas  += parallelSequences * effLayerSize;
            }

            int numPatterns = patternsCount;
            if (skipFirstPattern || skipLastPattern)
                numPatterns -= parallelSequences;

            real_t wu = 0;
            for (int i = 0; i < numPatterns; ++i) {
                wu += (offOutputs ? *offOutputs : bias) * *offDeltas;
                    
                offOutputs += offOutputsInc;
                offDeltas  += effLayerSize;
            }

            return wu;
        }
    };
    
} // anonymous namespace
} // namespace internal


namespace layers {

    template <typename TDevice>
    LstmLayer<TDevice>::LstmLayer(const helpers::JsonValue &layerChild, 
                                  const helpers::JsonValue &weightsSection,
                                  Layer<TDevice> &precedingLayer,
                                  bool bidirectional)
        : TrainableLayer<TDevice>(layerChild, weightsSection, 4, (bidirectional ? 2 : 4) * helpers::safeJsonGetInt(layerChild, "size") + 3, precedingLayer)
        , m_isBidirectional      (bidirectional)
    {
        if (m_isBidirectional && this->size() % 2 != 0)
            throw std::runtime_error("Cannot create a bidirectional layer with an odd layer size");

        // set raw pointers
        int ls  = this->size();
        int pls = this->precedingLayer().size();

        _rawNiBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 0 * ls;
        _rawIgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 1 * ls;
        _rawFgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 2 * ls;
        _rawOgBiasWeights     = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 3 * ls;
        _rawIgPeepholeWeights = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls + 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 0 * ls;
        _rawFgPeepholeWeights = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls + 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 1 * ls;
        _rawOgPeepholeWeights = helpers::getRawPointer(this->weights()) + 4 * ls * pls + 4 * ls + 4 * ls * ls / (m_isBidirectional ? 2 : 1) + 2 * ls;

        // create the forward and backward info structs
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            // calculate sizes
            int pls = this->precedingLayer().size();
            int ls  = this->size();
            int els = this->size() / (m_isBidirectional ? 2 : 1);

            // cell states, niags, deltas, ...
            Cpu::real_vector tmp(this->outputs().size() / (m_isBidirectional ? 2 : 1), 0);

            if (m_isBidirectional) {
                fwbw->tmpOutputs      = tmp;
                fwbw->tmpOutputErrors = tmp;
            }
            else {
                fwbw->tmpOutputs     .swap(this->_outputs());
                fwbw->tmpOutputErrors.swap(this->outputErrors());
            }

            fwbw->cellStates      = tmp;
            fwbw->cellStateErrors = tmp;
            fwbw->niActs          = tmp;
            fwbw->igActs          = tmp;
            fwbw->fgActs          = tmp;
            fwbw->ogActs          = tmp;
            fwbw->niDeltas        = tmp;
            fwbw->igDeltas        = tmp;
            fwbw->fgDeltas        = tmp;
            fwbw->ogDeltas        = tmp;

            // weight matrices
            weight_matrices_t* wmArr [] = { &fwbw->weightMatrices, &fwbw->weightUpdateMatrices };
            real_vector*       wtsArr[] = { &this->weights(),      &this->_weightUpdates() };
            for (int wmArrIdx = 0; wmArrIdx < 2; ++wmArrIdx) {
                weight_matrices_t *wm  = wmArr [wmArrIdx];
                real_vector       *wts = wtsArr[wmArrIdx];

                int numInputWeights      = ls * pls;
                int numInternalWeights   = ls * els;
                int inputWeightsStart    = ((fwbwArrIdx == 1) ? (numInputWeights    / 2) : 0);
                int internalWeightsStart = ((fwbwArrIdx == 1) ? (numInternalWeights / 2) : 0) + 4 * (ls * (pls + 1));

                wm->niInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 0 * numInputWeights);
                wm->igInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 1 * numInputWeights);
                wm->fgInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 2 * numInputWeights);
                wm->ogInput = helpers::Matrix<TDevice>(wts, pls, els, inputWeightsStart + 3 * numInputWeights);

                wm->niInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 0 * numInternalWeights);
                wm->igInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 1 * numInternalWeights);
                wm->fgInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 2 * numInternalWeights);
                wm->ogInternal = helpers::Matrix<TDevice>(wts, els, els, internalWeightsStart + 3 * numInternalWeights);
            }

            // matrices for each timestep
            for (int timestep = 0; timestep < this->maxSeqLength(); ++timestep) {
                int rows   = this->size() / (m_isBidirectional ? 2 : 1);
                int cols   = this->parallelSequences();
                int offset = timestep * rows * cols;

                timestep_matrices_t tm;
                tm.tmpOutputs      = helpers::Matrix<TDevice>(&fwbw->tmpOutputs,      rows, cols, offset);
                tm.tmpOutputErrors = helpers::Matrix<TDevice>(&fwbw->tmpOutputErrors, rows, cols, offset);
                tm.niActs          = helpers::Matrix<TDevice>(&fwbw->niActs,          rows, cols, offset);
                tm.igActs          = helpers::Matrix<TDevice>(&fwbw->igActs,          rows, cols, offset);
                tm.fgActs          = helpers::Matrix<TDevice>(&fwbw->fgActs,          rows, cols, offset);
                tm.ogActs          = helpers::Matrix<TDevice>(&fwbw->ogActs,          rows, cols, offset);
                tm.niDeltas        = helpers::Matrix<TDevice>(&fwbw->niDeltas,        rows, cols, offset);
                tm.igDeltas        = helpers::Matrix<TDevice>(&fwbw->igDeltas,        rows, cols, offset);
                tm.fgDeltas        = helpers::Matrix<TDevice>(&fwbw->fgDeltas,        rows, cols, offset);
                tm.ogDeltas        = helpers::Matrix<TDevice>(&fwbw->ogDeltas,        rows, cols, offset);

                fwbw->timestepMatrices.push_back(tm);
            }
        }

        if (!m_isBidirectional) {
            m_fw.tmpOutputs     .swap(this->_outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }
    }

    template <typename TDevice>
    LstmLayer<TDevice>::~LstmLayer()
    {
    }

    template <typename TDevice>
    const std::string& LstmLayer<TDevice>::type() const
    {
        static const std::string su("lstm");
        static const std::string sb("blstm");
        return (m_isBidirectional ? sb : su);
    }

    template <typename TDevice>
    bool LstmLayer<TDevice>::isBidirectional() const
    {
        return m_isBidirectional;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::cellStates() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStates;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::cellStateErrors() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.cellStateErrors;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::netInputActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::netInputDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.niDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::inputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::inputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.igDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::forgetGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::forgetGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.fgDeltas;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::outputGateActs() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogActs;
    }

    template <typename TDevice>
    const typename TDevice::real_vector& LstmLayer<TDevice>::outputGateDeltas() const
    {
        if (m_isBidirectional)
            throw std::runtime_error("Not implemented");
        else
            return m_fw.ogDeltas;
    }

    template <typename TDevice>
    void LstmLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
    {
        TrainableLayer<TDevice>::loadSequences(fraction);

        m_precLayerOutputsMatrix = helpers::Matrix<TDevice>(&this->precedingLayer().outputs(), this->precedingLayer().size(), this->curMaxSeqLength() * this->parallelSequences());

        // update the niag matrices
        forward_backward_info_t* fwbwArr[] = { &m_fw, &m_bw };
        for (int fwbwArrIdx = 0; fwbwArrIdx < (m_isBidirectional ? 2 : 1); ++fwbwArrIdx) {
            forward_backward_info_t *fwbw = fwbwArr[fwbwArrIdx];

            int rows = this->size() / (m_isBidirectional ? 2 : 1);
            int cols = this->curMaxSeqLength() * this->parallelSequences();

            fwbw->niActsMatrix = helpers::Matrix<TDevice>(&fwbw->niActs, rows, cols);
            fwbw->igActsMatrix = helpers::Matrix<TDevice>(&fwbw->igActs, rows, cols);
            fwbw->fgActsMatrix = helpers::Matrix<TDevice>(&fwbw->fgActs, rows, cols);
            fwbw->ogActsMatrix = helpers::Matrix<TDevice>(&fwbw->ogActs, rows, cols);

            fwbw->niDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->niDeltas, rows, cols);
            fwbw->igDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->igDeltas, rows, cols);
            fwbw->fgDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->fgDeltas, rows, cols);
            fwbw->ogDeltasMatrix = helpers::Matrix<TDevice>(&fwbw->ogDeltas, rows, cols);
        }
    }

    template <typename TDevice>
    void LstmLayer<TDevice>::computeForwardPass()
    {
        // for unidirectional LSTM, we can write the outputs directly in the layer output vector
        if (!m_isBidirectional) {
            m_fw.tmpOutputs.swap(this->_outputs());
        }

        // sum up the activations from the preceding layer
        {{
            // forward states
            m_fw.niActsMatrix.assignProduct(m_fw.weightMatrices.niInput, true, m_precLayerOutputsMatrix, false);
            m_fw.igActsMatrix.assignProduct(m_fw.weightMatrices.igInput, true, m_precLayerOutputsMatrix, false);
            m_fw.fgActsMatrix.assignProduct(m_fw.weightMatrices.fgInput, true, m_precLayerOutputsMatrix, false);
            m_fw.ogActsMatrix.assignProduct(m_fw.weightMatrices.ogInput, true, m_precLayerOutputsMatrix, false);

            // backward states
            if (m_isBidirectional) {
                m_bw.niActsMatrix.assignProduct(m_bw.weightMatrices.niInput, true, m_precLayerOutputsMatrix, false);
                m_bw.igActsMatrix.assignProduct(m_bw.weightMatrices.igInput, true, m_precLayerOutputsMatrix, false);
                m_bw.fgActsMatrix.assignProduct(m_bw.weightMatrices.fgInput, true, m_precLayerOutputsMatrix, false);
                m_bw.ogActsMatrix.assignProduct(m_bw.weightMatrices.ogInput, true, m_precLayerOutputsMatrix, false);
            }
        }}

        // compute the block outputs
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockOutputFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.bias               = this->bias();
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.niBiasWeights      = _rawNiBiasWeights;
            fn.igBiasWeights      = _rawIgBiasWeights;
            fn.fgBiasWeights      = _rawFgBiasWeights;
            fn.ogBiasWeights      = _rawOgBiasWeights;
            fn.igPeepWeights      = _rawIgPeepholeWeights;
            fn.fgPeepWeights      = _rawFgPeepholeWeights;
            fn.ogPeepWeights      = _rawOgPeepholeWeights;
            fn.cellStates         = helpers::getRawPointer(m_fw.cellStates);
            fn.niActs             = helpers::getRawPointer(m_fw.niActs);
            fn.igActs             = helpers::getRawPointer(m_fw.igActs);
            fn.fgActs             = helpers::getRawPointer(m_fw.fgActs);
            fn.ogActs             = helpers::getRawPointer(m_fw.ogActs);

            for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                // collect outputs from previous timestep
                if (timestep != 0) {
                    m_fw.timestepMatrices[timestep].niActs.addProduct(m_fw.weightMatrices.niInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                    m_fw.timestepMatrices[timestep].igActs.addProduct(m_fw.weightMatrices.igInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                    m_fw.timestepMatrices[timestep].fgActs.addProduct(m_fw.weightMatrices.fgInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                    m_fw.timestepMatrices[timestep].ogActs.addProduct(m_fw.weightMatrices.ogInternal, true, m_fw.timestepMatrices[timestep-1].tmpOutputs, false);
                }

                // compute outputs
                thrust::transform(
                    thrust::counting_iterator<int>(n*timestep),
                    thrust::counting_iterator<int>(n*timestep) + n,
                    thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<bool>(!timestep), thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    m_fw.tmpOutputs.begin() + n*timestep,
                    fn
                    );
            }

            // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.niBiasWeights     += els;
                fn.igBiasWeights     += els;
                fn.fgBiasWeights     += els;
                fn.ogBiasWeights     += els;
                fn.igPeepWeights     += els;
                fn.fgPeepWeights     += els;
                fn.ogPeepWeights     += els;
                fn.cellStates         = helpers::getRawPointer(m_bw.cellStates);
                fn.niActs             = helpers::getRawPointer(m_bw.niActs);
                fn.igActs             = helpers::getRawPointer(m_bw.igActs);
                fn.fgActs             = helpers::getRawPointer(m_bw.fgActs);
                fn.ogActs             = helpers::getRawPointer(m_bw.ogActs);

                for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                    // collect outputs from previous timestep
                    if (timestep != this->curMaxSeqLength()-1) {
                        m_bw.timestepMatrices[timestep].niActs.addProduct(m_bw.weightMatrices.niInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].igActs.addProduct(m_bw.weightMatrices.igInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].fgActs.addProduct(m_bw.weightMatrices.fgInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                        m_bw.timestepMatrices[timestep].ogActs.addProduct(m_bw.weightMatrices.ogInternal, true, m_bw.timestepMatrices[timestep+1].tmpOutputs, false);
                    }

                    // compute outputs
                    thrust::transform(
                        thrust::counting_iterator<int>(n*timestep),
                        thrust::counting_iterator<int>(n*timestep) + n,
                        thrust::make_zip_iterator(thrust::make_tuple(thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1), thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                        m_bw.tmpOutputs.begin() + n*timestep,
                        fn
                        );
                }
            }
        }}

        // resort outputs
        if (m_isBidirectional) {
            internal::ResortOutputsFn fn;
            fn.layerSize    = this->size();
            fn.effLayerSize = this->size() / 2;
            fn.fwOutputs    = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs    = helpers::getRawPointer(m_bw.tmpOutputs);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + this->curMaxSeqLength() * this->parallelSequences() * this->size(),
                this->_outputs().begin(),
                fn
                );
        }
        else {
            this->_outputs().swap(m_fw.tmpOutputs);
        }
    }

    template <typename TDevice>
    void LstmLayer<TDevice>::computeBackwardPass()
    {
        // for unidirectional LSTM, we can write the output errors directly in the layer output errors vector
        if (m_isBidirectional) {
            internal::ResortOutputErrorsFn fn;
            fn.layerSize      = this->size();
            fn.effLayerSize   = this->size() / 2;
            fn.fwOutputErrors = helpers::getRawPointer(m_fw.tmpOutputErrors);
            fn.bwOutputErrors = helpers::getRawPointer(m_bw.tmpOutputErrors);

            int n = this->curMaxSeqLength() * this->parallelSequences() * this->size();

            thrust::for_each(
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin(),   thrust::counting_iterator<int>(0))),
                thrust::make_zip_iterator(thrust::make_tuple(this->outputErrors().begin()+n, thrust::counting_iterator<int>(0)+n)),
                fn
                );
        }
        else {
            m_fw.tmpOutputs     .swap(this->outputs());
            m_fw.tmpOutputErrors.swap(this->outputErrors());
        }

        // calculate the block errors
        {{
            int els = this->size() / (m_isBidirectional ? 2 : 1);
            int n   = this->parallelSequences() * els;

            // forward states
            internal::ComputeBlockErrorsFn fn;
            fn.effLayerSize       = els;
            fn.prevOutputDistance = -n;
            fn.patTypes           = helpers::getRawPointer(this->patTypes());
            fn.igPeepWeights      = _rawIgPeepholeWeights;
            fn.fgPeepWeights      = _rawFgPeepholeWeights;
            fn.ogPeepWeights      = _rawOgPeepholeWeights;
            fn.cellStates         = helpers::getRawPointer(m_fw.cellStates);
            fn.niActs             = helpers::getRawPointer(m_fw.niActs);
            fn.igActs             = helpers::getRawPointer(m_fw.igActs);
            fn.fgActs             = helpers::getRawPointer(m_fw.fgActs);
            fn.ogActs             = helpers::getRawPointer(m_fw.ogActs);
            fn.cellStateErrors    = helpers::getRawPointer(m_fw.cellStateErrors);
            fn.niDeltas           = helpers::getRawPointer(m_fw.niDeltas);
            fn.igDeltas           = helpers::getRawPointer(m_fw.igDeltas);
            fn.fgDeltas           = helpers::getRawPointer(m_fw.fgDeltas);
            fn.ogDeltas           = helpers::getRawPointer(m_fw.ogDeltas);

            for (int timestep = this->curMaxSeqLength()-1; timestep >= 0; --timestep) {
                // collect errors from previous timestep
                if (timestep != this->curMaxSeqLength()-1) {
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.niInternal, false, m_fw.timestepMatrices[timestep+1].niDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.igInternal, false, m_fw.timestepMatrices[timestep+1].igDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.fgInternal, false, m_fw.timestepMatrices[timestep+1].fgDeltas, false);
                    m_fw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_fw.weightMatrices.ogInternal, false, m_fw.timestepMatrices[timestep+1].ogDeltas, false);
                }

                // compute errors
                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(m_fw.tmpOutputErrors.begin() + n*timestep,   thrust::counting_iterator<int>(n*timestep),   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),   thrust::constant_iterator<bool>(!timestep),   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                    thrust::make_zip_iterator(thrust::make_tuple(m_fw.tmpOutputErrors.begin() + n*timestep+n, thrust::counting_iterator<int>(n*timestep)+n, thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n, thrust::constant_iterator<bool>(!timestep)+n, thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                    fn
                    );
            }

            // backward states
            if (m_isBidirectional) {
                fn.prevOutputDistance = +n;
                fn.igPeepWeights     += els;
                fn.fgPeepWeights     += els;
                fn.ogPeepWeights     += els;
                fn.cellStates         = helpers::getRawPointer(m_bw.cellStates);
                fn.niActs             = helpers::getRawPointer(m_bw.niActs);
                fn.igActs             = helpers::getRawPointer(m_bw.igActs);
                fn.fgActs             = helpers::getRawPointer(m_bw.fgActs);
                fn.ogActs             = helpers::getRawPointer(m_bw.ogActs);
                fn.cellStateErrors    = helpers::getRawPointer(m_bw.cellStateErrors);
                fn.niDeltas           = helpers::getRawPointer(m_bw.niDeltas);
                fn.igDeltas           = helpers::getRawPointer(m_bw.igDeltas);
                fn.fgDeltas           = helpers::getRawPointer(m_bw.fgDeltas);
                fn.ogDeltas           = helpers::getRawPointer(m_bw.ogDeltas);

                for (int timestep = 0; timestep < this->curMaxSeqLength(); ++timestep) {
                    // collect errors from previous timestep
                    if (timestep != 0) {
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.niInternal, false, m_bw.timestepMatrices[timestep-1].niDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.igInternal, false, m_bw.timestepMatrices[timestep-1].igDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.fgInternal, false, m_bw.timestepMatrices[timestep-1].fgDeltas, false);
                        m_bw.timestepMatrices[timestep].tmpOutputErrors.addProduct(m_bw.weightMatrices.ogInternal, false, m_bw.timestepMatrices[timestep-1].ogDeltas, false);
                    }

                    // compute errors
                    thrust::for_each(
                        thrust::make_zip_iterator(thrust::make_tuple(m_bw.tmpOutputErrors.begin() + n*timestep,   thrust::counting_iterator<int>(n*timestep),   thrust::constant_iterator<bool>(!timestep),   thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1),   thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength()))),
                        thrust::make_zip_iterator(thrust::make_tuple(m_bw.tmpOutputErrors.begin() + n*timestep+n, thrust::counting_iterator<int>(n*timestep)+n, thrust::constant_iterator<bool>(!timestep)+n, thrust::constant_iterator<bool>(timestep == this->curMaxSeqLength()-1)+n, thrust::constant_iterator<bool>(timestep >= this->curMinSeqLength())+n)),
                        fn
                        );
                }
            }
        }}

        // back-propagate the error to the preceding layer
        {{
            TrainableLayer<TDevice> *pl = dynamic_cast<TrainableLayer<TDevice>*>(&this->precedingLayer());
            if (pl) {
                helpers::Matrix<TDevice> plErrorsMatrix(&pl->outputErrors(), pl->size(), this->curMaxSeqLength() * this->parallelSequences());

                // forward states
                plErrorsMatrix.assignProduct(m_fw.weightMatrices.niInput, false, m_fw.niDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.igInput, false, m_fw.igDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.fgInput, false, m_fw.fgDeltasMatrix, false);
                plErrorsMatrix.addProduct   (m_fw.weightMatrices.ogInput, false, m_fw.ogDeltasMatrix, false);

                // backward states
                if (m_isBidirectional) {
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.niInput, false, m_bw.niDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.igInput, false, m_bw.igDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.fgInput, false, m_bw.fgDeltasMatrix, false);
                    plErrorsMatrix.addProduct(m_bw.weightMatrices.ogInput, false, m_bw.ogDeltasMatrix, false);
                }
            }
        }}

        // compute the weight updates
        {{
            internal::ComputeWeightUpdateFn fn;
            fn.layerSize             = this->size();
            fn.effLayerSize          = this->size() / (m_isBidirectional ? 2 : 1);
            fn.precLayerSize         = this->precedingLayer().size();
            fn.timestepDistance      = this->parallelSequences() * this->size() / (m_isBidirectional ? 2 : 1);
            fn.parallelSequences     = this->parallelSequences();
            fn.patternsCount         = this->curMaxSeqLength() * this->parallelSequences();
            fn.biasWeightsOffset     = this->size() * this->precedingLayer().size() * 4;
            fn.internalWeightsOffset = fn.biasWeightsOffset + this->size() * 4;
            fn.peepholeWeightsOffset = fn.internalWeightsOffset + this->size() * fn.effLayerSize * 4;
            fn.bias                  = this->bias();
            fn.plOutputs             = helpers::getRawPointer(this->precedingLayer().outputs());
            fn.fwOutputs             = helpers::getRawPointer(m_fw.tmpOutputs);
            fn.bwOutputs             = helpers::getRawPointer(m_bw.tmpOutputs);
            fn.fwCellStates          = helpers::getRawPointer(m_fw.cellStates);
            fn.bwCellStates          = helpers::getRawPointer(m_bw.cellStates);
            fn.fwNiDeltas            = helpers::getRawPointer(m_fw.niDeltas);
            fn.bwNiDeltas            = helpers::getRawPointer(m_bw.niDeltas);
            fn.fwIgDeltas            = helpers::getRawPointer(m_fw.igDeltas);
            fn.bwIgDeltas            = helpers::getRawPointer(m_bw.igDeltas);
            fn.fwFgDeltas            = helpers::getRawPointer(m_fw.fgDeltas);
            fn.bwFgDeltas            = helpers::getRawPointer(m_bw.fgDeltas);
            fn.fwOgDeltas            = helpers::getRawPointer(m_fw.ogDeltas);
            fn.bwOgDeltas            = helpers::getRawPointer(m_bw.ogDeltas);

            thrust::transform(
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(0) + (int)this->weightUpdates().size(),
                this->_weightUpdates().begin(),
                fn
                );
        }}

        // re-swap the output errors and the tmp output errors of the forward pass
        if (!m_isBidirectional) {
            this->outputErrors().swap(m_fw.tmpOutputErrors);
            this->_outputs()    .swap(m_fw.tmpOutputs);
        }
    }


    // explicit template instantiations
    template class LstmLayer<Cpu>;
    template class LstmLayer<Gpu>;

} // namespace layers
