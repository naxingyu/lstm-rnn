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

#ifndef TYPES_HPP
#define TYPES_HPP

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#define PATTYPE_NONE   0 ///< pattern does not belong to the sequence
#define PATTYPE_FIRST  1 ///< first pattern/timestep in the sequence
#define PATTYPE_NORMAL 2 ///< pattern/timestep with a sequence (not first/last)
#define PATTYPE_LAST   3 ///< last pattern/timestep in the sequence


/*************************************************************************//**
 * The floating point type used for all computations
 *****************************************************************************/
typedef float real_t;


/*************************************************************************//**
 * Data types on the CPU
 *****************************************************************************/
struct Cpu
{
    enum { cublas_capable = false };

    typedef thrust::host_vector<real_t> real_vector;
    typedef thrust::host_vector<int>    int_vector;
    typedef thrust::host_vector<bool>   bool_vector;
    typedef thrust::host_vector<char>   pattype_vector;
};


/*************************************************************************//**
 * Data types on the GPU
 *****************************************************************************/
struct Gpu
{
    enum { cublas_capable = true };

    typedef thrust::device_vector<real_t> real_vector;
    typedef thrust::device_vector<int>    int_vector;
    typedef thrust::device_vector<bool>   bool_vector;
    typedef thrust::device_vector<char>   pattype_vector;
};


#endif // TYPES_HPP
