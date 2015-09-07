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
#include <networkserializer.h>
#include <dense.h>
#include <gru.h>
#include <lstm.h>
#include <cwrnn.h>

#include <iostream>

inline Network *makeGRU(unsigned int nin, unsigned int nhidden, unsigned int nout, float learning_rate)
{
    Network *net = new Network(nin);
    Dense *dense_in = new Dense(nhidden, learning_rate);
    Dense *dense_z = new Dense(nhidden, learning_rate);
    Dense *dense_r = new Dense(nhidden, learning_rate);
    GRU *gru = new GRU(nhidden, learning_rate);
    Dense *out = new Dense(nout, learning_rate);

    dense_in->setInput(net->inputPort());
    dense_z->setInput(net->inputPort());
    dense_r->setInput(net->inputPort());
    gru->addInput(dense_in->output());
    gru->addZ(dense_z->output());
    gru->addR(dense_r->output());
    out->setInput(gru->output());

    net->addNode(dense_in);
    net->addNode(dense_z);
    net->addNode(dense_r);
    net->addNode(gru);
    net->addNode(out);

    return net;
}

inline Network *makeLSTM(unsigned int nin, unsigned int nhidden, unsigned int nout, float learning_rate)
{
    Network *net = new Network(nin);
    Dense *dense_in = new Dense(nhidden, learning_rate);
    Dense *dense_ingate = new Dense(nhidden, learning_rate);
    Dense *dense_outgate = new Dense(nhidden, learning_rate);
    Dense *dense_forgetgate = new Dense(nhidden, learning_rate);
    LSTM *lstm = new LSTM(nhidden, learning_rate);
    Dense *out = new Dense(nout, learning_rate);

    dense_in->setInput(net->inputPort());
    dense_ingate->setInput(net->inputPort());
    dense_outgate->setInput(net->inputPort());
    dense_forgetgate->setInput(net->inputPort());
    lstm->addInput(dense_in->output());
    lstm->addInGate(dense_ingate->output());
    lstm->addOutGate(dense_outgate->output());
    lstm->addForgetGate(dense_forgetgate->output());
    out->setInput(lstm->output());

    net->addNode(dense_in);
    net->addNode(dense_ingate);
    net->addNode(dense_outgate);
    net->addNode(dense_forgetgate);
    net->addNode(lstm);
    net->addNode(out);

    return net;
}

inline Network *makeCWRNN(unsigned int num_units, unsigned int nin, unsigned int nhidden, unsigned int nout, float learning_rate)
{
    Network *net = new Network(nin);
    CWRNN *cwrnn = new CWRNN(num_units, nhidden, learning_rate);
    Dense *out = new Dense(nout, learning_rate);

    cwrnn->addInput(net->inputPort());
    out->setInput(cwrnn->output());

    net->addNode(cwrnn);
    net->addNode(out);

    return net;
}

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
 * @brief Make a sequence of vectors from a sequence of floats, each vector having
 *        a size of one.
 */
inline std::vector<Vector> makeSequence(const std::vector<Float> &entries)
{
    std::vector<Vector> rs;

    for (Float entry : entries) {
        rs.push_back(makeVector({entry}));
    }

    return rs;
}

/**
 * @brief Train a network on input data and return its training error
 */
inline float trainNetwork(Network *network,
                          const std::vector<Vector> &input,
                          const std::vector<Vector> &output,
                          unsigned int iterations,
                          bool verbose = true,
                          bool sequence = false,
                          bool test_serialize = false)
{
    Eigen::MatrixXf inputs(input[0].rows(), input.size());
    Eigen::MatrixXf outputs(output[0].rows(), output.size());

    // Copy the vectors into their matrices
    for (std::size_t i=0; i<input.size(); ++i) {
        inputs.col(i) = input[i];
        outputs.col(i) = output[i];
    }

    // Train the network
    if (sequence) {
        network->trainSequence(inputs, outputs, iterations);
    } else {
        network->train(inputs, outputs, 1, iterations);
    }

    if (test_serialize) {
        // Serialize then deserialize the network (this tests serialization)
        NetworkSerializer serializer;

        network->serialize(serializer);
        network->deserialize(serializer);
    }

    // Check that learning was correct
    Float mse = 0.0f;

    network->reset();

    for (std::size_t i=0; i<input.size(); ++i) {
        if (sequence) {
            // Tell the network which time-step the prediction is for
            network->setCurrentTimestep(i);
        }

        // Make the prediction and compute errors
        Vector v = network->predict(input[i]);

        mse += (v.array() - output[i].array()).square().mean();

        if (verbose) {
            std::cout << input[i] << ' ' << v << ' ' << output[i] << std::endl;
        }
    }

    mse /= float(input.size());

    if (verbose) {
        std::cout << "Final MSE: " << mse << std::endl;
    }

    return mse;
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
                          bool sequence = false)
{
    return trainNetwork(network, input, output, iterations, verbose, sequence, true) < target_mse;
}

#endif
