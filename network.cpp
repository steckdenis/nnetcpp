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

#include "network.h"

#include <assert.h>
#include <algorithm>

Network::Network(unsigned int inputs)
{
    _input_port.error = Vector::Zero(inputs);
    _input_port.value = Vector::Zero(inputs);
}

AbstractNode::Port *Network::inputPort()
{
    return &_input_port;
}

AbstractNode::Port *Network::output()
{
    return _nodes.back()->output();
}

void Network::reset()
{
    // Call reset on all the nodes
    AbstractRecurrentNetworkNode::reset();

    // Set timestep to zero so that possible recurrent nodes work as expected
    setCurrentTimestep(0);
}

void Network::clearError()
{
    // Clear the error of all the nodes
    AbstractRecurrentNetworkNode::clearError();

    // Clear the error of the input port, in case it is used somewhere.
    _input_port.error.setZero();
}

void Network::update()
{
    // Tell all the nodes to update their parameters
    AbstractRecurrentNetworkNode::update();

    clearError();
}

void Network::train(const Eigen::MatrixXf &inputs,
                    const Eigen::MatrixXf &outputs,
                    unsigned int batch_size,
                    unsigned int epochs)
{
    train(inputs, outputs, nullptr, batch_size, epochs);
}

void Network::trainSequence(const Eigen::MatrixXf &inputs,
                            const Eigen::MatrixXf &outputs,
                            unsigned int epochs)
{
    trainSequence(inputs, outputs, nullptr, epochs);
}

void Network::train(const Eigen::MatrixXf &inputs,
                    const Eigen::MatrixXf &outputs,
                    const Eigen::MatrixXf &weights,
                    unsigned int batch_size,
                    unsigned int epochs)
{
    train(inputs, outputs, &weights, batch_size, epochs);
}

void Network::trainSequence(const Eigen::MatrixXf &inputs,
                            const Eigen::MatrixXf &outputs,
                            const Eigen::MatrixXf &weights,
                            unsigned int epochs)
{
    trainSequence(inputs, outputs, &weights, epochs);
}

void Network::train(const Eigen::MatrixXf &inputs,
                    const Eigen::MatrixXf &outputs,
                    const Eigen::MatrixXf *weights,
                    unsigned int batch_size,
                    unsigned int epochs)
{
    std::vector<int> indexes(inputs.cols());

    for (int i=0; i<inputs.cols(); ++i) {
        indexes[i] = i;
    }

    // Place the network at time step zero (it may contain recurrent nodes that
    // crash if this is not called)
    reset();

    // Epochs
    for (unsigned int epoch=0; epoch < epochs; ++epoch) {
        // Shuffle the input vectors to improve non-sequence learning
        std::random_shuffle(indexes.begin(), indexes.end());

        // Perform the training
        unsigned int batch_remaining = batch_size;

        for (int index : indexes) {
            predict(inputs.col(index), nullptr);

            if (weights == nullptr) {
                setExpectedOutput(outputs.col(index));
            } else {
                setExpectedOutput(outputs.col(index), weights->col(index));
            }

            if (--batch_remaining == 0) {
                batch_remaining = batch_size;

                update();
            }
        }
    }
}

void Network::trainSequence(const Eigen::MatrixXf &inputs,
                            const Eigen::MatrixXf &outputs,
                            const Eigen::MatrixXf *weights,
                            unsigned int epochs)
{
    Eigen::MatrixXf errors(outputs.rows(), outputs.cols());

    // Reset the network before any learning
    reset();

    // Epochs
    for (unsigned int epoch=0; epoch < epochs; ++epoch) {
        // Forward pass in the network, store the errors in a matrix
        for (int t=0; t<outputs.cols(); ++t) {
            setCurrentTimestep(t);
            predict(inputs.col(t), nullptr);

            if (weights == nullptr) {
                errors.col(t) = outputs.col(t) - output()->value;
            } else {
                errors.col(t) = (outputs.col(t) - output()->value).cwiseProduct(weights->col(t));
            }
        }

        // Now that the errors through time are computed, the backward pass can be
        // performed
        for (int t=outputs.cols()-1; t>=0; --t) {
            // Rewind the network to the previous time step, and re-forward
            // it. This allows the network to recover its internal state from
            // time t-1, which will allow setError (next loop iteration) to
            // behave correctly
            setCurrentTimestep(t);
            predict(inputs.col(t));

            // Set the error at the output of the network and backpropagate it.
            // Some nodes (GRU, LSTM) will also backpropagate error from t+1 to t.
            setError(errors.col(t));
        }

        update();
        reset();
    }
}
