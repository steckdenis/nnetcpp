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
                          bool verbose = true)
{
    Float mse;

    // Perform the iterations
    for (unsigned int iteration=0; iteration<iterations; ++iteration) {
        // Train the network over all the input/output samples
        mse = 0.0f;

        for (std::size_t i=0; i<input.size(); ++i) {
            mse += network->trainSample(input[i], output[i]);
        }

        network->reset();

        // Stats
        mse /= Float(input.size());

        if (verbose) {
            std::cout << "Iteration " << iteration << ": mse = " << mse << std::endl;
        }

        if (mse < target_mse) {
            // Learning was possible
            break;
        }
    }

    std::cout << "Final MSE = " << mse << std::endl;

    // No learning possible, print the values for debugging
    if (verbose) {
        for (std::size_t i=0; i<input.size(); ++i) {
            Vector v = network->predict(input[i]);

            std::cout << input[i] << ' ' << v << ' ' << output[i] << std::endl;
        }
    }

    return mse < target_mse;   // No learning possible
}

#endif