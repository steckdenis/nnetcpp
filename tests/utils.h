#ifndef __UTILS_H__
#define __UTILS_H__

#include <abstractnode.h>
#include <network.h>
#include <iostream>

/**
 * @brief Make a vector from a sequence of floats
 */
Vector makeVector(const std::vector<Float> data)
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
bool checkLearning(Network *network,
                   const std::vector<Vector> input,
                   const std::vector<Vector> output,
                   Float target_mse,
                   unsigned int iterations)
{
    Float mse;

    for (unsigned int iteration=0; iteration<iterations; ++iteration) {
        // Train the network over all the input/output samples
        mse = 0.0f;

        for (std::size_t i=0; i<input.size(); ++i) {
            mse += network->trainSample(input[i], output[i]);
        }

        // Stats
        mse /= Float(input.size());

        std::cout << "Iteration " << iteration << ": mse = " << mse << std::endl;

        if (mse < target_mse) {
            // Learning was possible
            return true;
        }
    }

    return false;   // No learning possible
}

#endif