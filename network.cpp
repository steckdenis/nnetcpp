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

Network::~Network()
{
    for (AbstractNode *node : _nodes) {
        delete node;
    }
}

AbstractNode::Port* Network::inputPort()
{
    return &_input_port;
}

void Network::addNode(AbstractNode *node)
{
    _nodes.push_back(node);
}

Vector Network::predict(const Vector &input)
{
    assert(input.rows() == _input_port.value.rows());

    // Put the input in the input port, and propagate it through the network
    _input_port.value = input;

    for (AbstractNode *node : _nodes) {
        node->forward();
    }

    // Return the output of the last node
    AbstractNode *last = _nodes.back();

    return last->output()->value;
}

void Network::reset()
{
    // Call reset on all the nodes
    for (AbstractNode *node : _nodes) {
        node->reset();
    }
}

Float Network::setExpectedOutput(const Vector &output)
{
    return setExpectedOutput(output, nullptr);
}

Float Network::setExpectedOutput(const Vector &output, const Vector &weights)
{
    return setExpectedOutput(output, &weights);
}

Float Network::setExpectedOutput(const Vector &output, const Vector *weights)
{
    AbstractNode *last = _nodes.back();

    if (weights == nullptr) {
        return setError(output - last->output()->value);
    } else {
        return setError((output - last->output()->value).cwiseProduct(*weights));
    }
}

Float Network::setError(const Vector &error)
{
    AbstractNode *last = _nodes.back();

    // Set the error of the last node
    assert(error.rows() == last->output()->error.rows());

    last->output()->error = error;

    // Backpropagate it
    for (int i=_nodes.size()-1; i>=0; --i) {
        _nodes[i]->backward();
    }

    return error.array().square().mean();
}

void Network::update()
{
    // Tell all the nodes to update their parameters, then discard the error signal
    for (AbstractNode *node : _nodes) {
        node->update();
        node->clearError();
    }

    _input_port.error.setZero();
}

Float Network::trainSample(const Vector &input, const Vector &output)
{
    return trainSample(input, output, nullptr);
}

Float Network::trainSample(const Vector &input, const Vector &output, const Vector &weights)
{
    return trainSample(input, output, &weights);
}

Float Network::trainSample(const Vector &input, const Vector &output, const Vector *weights)
{
    Float error;

    predict(input);
    error = setExpectedOutput(output, weights);
    update();

    return error;
}

void Network::train(const Eigen::MatrixXf &inputs,
                    const Eigen::MatrixXf &outputs,
                    unsigned int batch_size,
                    unsigned int epochs,
                    bool shuffle)
{
    train(inputs, outputs, nullptr, batch_size, epochs, shuffle);
}

void Network::train(const Eigen::MatrixXf &inputs,
                    const Eigen::MatrixXf &outputs,
                    const Eigen::MatrixXf &weights,
                    unsigned int batch_size,
                    unsigned int epochs,
                    bool shuffle)
{
    train(inputs, outputs, &weights, batch_size, epochs, shuffle);
}

void Network::train(const Eigen::MatrixXf &inputs,
                    const Eigen::MatrixXf &outputs,
                    const Eigen::MatrixXf *weights,
                    unsigned int batch_size,
                    unsigned int epochs,
                    bool shuffle)
{
    std::vector<int> indexes(inputs.cols());

    for (int i=0; i<inputs.cols(); ++i) {
        indexes[i] = i;
    }

    // Epochs
    for (unsigned int epoch=0; epoch < epochs; ++epoch) {
        // Shuffle the input vectors
        if (shuffle) {
            std::random_shuffle(indexes.begin(), indexes.end());
        }

        // Perform the training
        unsigned int batch_remaining = batch_size;

        for (int index : indexes) {
            predict(inputs.col(index));

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

        // Reset the network
        reset();
    }
}
