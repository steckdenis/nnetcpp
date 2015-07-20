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

#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "abstractnode.h"
#include <vector>

/**
 * @brief Neural network, made of nodes
 *
 * A Network object keeps track of nodes and manages the forward and backward
 * passes. It also contains useful methods for training.
 *
 * A neural network is built by first instantiating AbstractNode subclasses
 * and making connections (using their input and output ports). Then, the nodes
 * are added to the network in the order of forward propagation. If the network
 * is recurrent, this avoids loops. Usually, the breadth-first order is the one
 * to use for forward propagation.
 */
class Network
{
    public:
        /**
         * @param inputs Number of inputs of this network
         */
        Network(unsigned int inputs);
        ~Network();

        /**
         * @brief Port that will contain the inputs given to this network, so
         *        that the first node can read its input from somewhere
         */
        AbstractNode::Port *inputPort();

        /**
         * @brief Add a node to this network. The first node receives the input,
         *        the last one produces the output of the network.
         */
        void addNode(AbstractNode *node);

        /**
         * @brief Produce the output corresponding to the input.
         *
         * This prediction is incremental: recurrent networks are not reset between
         * calls to predict(). Ending an input sequence (and preparing the network
         * for the next one) is performed by reset().
         */
        Vector predict(const Vector &input);

        /**
         * @brief Clear the internal memory of the network but preserve its weights
         *
         * This method can be used between input sequences in order to clear the
         * internal memory of recurrent networks.
         */
        void reset();

        /**
         * @brief Set the expected output of this network and back-propagate the
         *        errors, without performing any gradient update.
         *
         * Calling predict() then setExpectedOutput() then update() allows to train
         * the network on one sample. Minibatches can be implemented by predicting
         * several values, calling setExpectedOutput() several times, then calling
         * update() one time.
         *
         * @return Mean squared error over the output neurons
         */
        Float setExpectedOutput(const Vector &output);

        /**
         * @brief Set the expected output of this network and compute the error
         *        using an error weight vector.
         * @sa train
         */
        Float setExpectedOutput(const Vector &output, const Vector &weights);

        /**
         * @brief Set the error signals at the output of this network and
         *        back-propagate it, without performing any gradient update
         *
         * @return Mean squared error over the output neurons
         */
        Float setError(const Vector &error);

        /**
         * @brief Perform one gradient update using the error computed by the
         *        last calls to setExpectedOutput() and setError()
         *
         * @note The error signal is zeroed-out after a call to this function
         */
        void update();

        /**
         * @brief Shortcut method that performs one gradient update on a sample
         *
         * @return Mean squared error over the output neurons
         */
        Float trainSample(const Vector &input, const Vector &output);

        /**
         * @brief Perform one gradient update for training, multiplying the error
         *        vector by @p weights
         *
         * This allows to define which output neurons have the more importance
         * when backpropagating the error.
         *
         * @return Mean squared error over the output neurons
         */
        Float trainSample(const Vector &input, const Vector &output, const Vector &weights);

        /**
         * @brief Train the network on a dataset
         *
         * @param inputs Matrix having one column per input vector
         * @param outputs Matrix having one column per output vector
         * @param batch_size Number of vectors handled before a gradient update is performed
         * @param epochs Number of epochs of training
         * @param shuffle True if the vectors are shuffled between the epochs
         *                (set to false if a sequence of observations has to be
         *                learned).
         */
        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   unsigned int batch_size,
                   unsigned int epochs,
                   bool shuffle);

        /**
         * @brief Train the network on a dataset, using weights for the output neurons
         *
         * @param weights Matrix having one column per weight vector. The weight
         *                vectors are used as described in trainSample().
         */
        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   const Eigen::MatrixXf &weights,
                   unsigned int batch_size,
                   unsigned int epochs,
                   bool shuffle);

    private:
        Float setExpectedOutput(const Vector &output, const Vector *weights);
        Float trainSample(const Vector &input, const Vector &output, const Vector *weights);
        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   const Eigen::MatrixXf *weights,
                   unsigned int batch_size,
                   unsigned int epochs,
                   bool shuffle);

    private:
        AbstractNode::Port _input_port;

        std::vector<AbstractNode *> _nodes;
};

#endif