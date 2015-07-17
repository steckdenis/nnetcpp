/*
 * Copyright (c) 2015 Vrije Universiteit Brussel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <abstractnode.h>
#include <network.h>

#include <iostream>

/**
 * @brief Make a vector from a sequence of floats
 */
inline Vector makeVector(const std::vector<Float> &data)
{
    Vector rs(data.size());

    for (std::size_t i=0; i<data.size(); ++i) {
        rs[i] = data[i];
    }

    return rs;
}

/**
 * @brief Check that a network manages to reduce its mean squared error below
 *        a threshold
 */
inline bool checkLearning(Network *network,
                          const std::vector<Vector> &input,
                          const std::vector<Vector> &output,
                          Float target_mse,
                          unsigned int iterations,
                          bool verbose = true,
                          bool shuffle = true)
{
    Eigen::MatrixXf inputs(input[0].rows(), input.size());
    Eigen::MatrixXf outputs(output[0].rows(), output.size());

    // Copy the vectors into their matrices
    for (std::size_t i=0; i<input.size(); ++i) {
        inputs.col(i) = input[i];
        outputs.col(i) = output[i];
    }

    // Train the network
    network->train(inputs, outputs, 1, iterations, shuffle);

    // Check that learning was correct
    Float mse = 0.0f;

    for (std::size_t i=0; i<input.size(); ++i) {
        Vector v = network->predict(input[i]);

        mse += (v.array() - output[i].array()).square().mean();

        if (verbose) {
            std::cout << input[i] << ' ' << v << ' ' << output[i] << std::endl;
        }
    }

    mse /= float(input.size());

    std::cout << "Final MSE: " << mse << std::endl;

    return mse < target_mse;   // No learning possible
}

#endif