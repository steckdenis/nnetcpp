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

#include "abstractrecurrentnetworknode.h"

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
 * to use for forward propagation. Recurrent nodes can be registered using addRecurrentNode
 * (they must also be added using addNode, like any node).
 *
 * @note When no fancy training methods are required, be sure to use train() and
 *       trainSequence() for learning (and predict() for prediction). Training
 *       a network is complicated and using setError, update and setTimestep
 *       has a great chance of giving bad results.
 */
class Network : public AbstractRecurrentNetworkNode
{
    public:
        /**
         * @param inputs Number of inputs of this network
         */
        Network(unsigned int inputs);

        /**
         * @brief Port that will contain the inputs given to this network, so
         *        that the first node can read its input from somewhere
         */
        Port *inputPort();

        /**
         * @brief Output port of the network, this is the output port of its last node
         */
        virtual Port *output();

        /**
         * @brief Produce the output corresponding to the input.
         *
         * This prediction is incremental: recurrent networks are not reset between
         * calls to predict(). Ending an input sequence (and preparing the network
         * for the next one) is performed by reset().
         *
         * @sa setTimestep()
         */
        Vector predict(const Vector &input);

        /**
         * @brief Clear the internal memory of the network but preserve its weights
         *
         * This method can be used between input sequences in order to clear the
         * internal memory of recurrent networks. It also resets the time-step to
         * zero.
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
         * @brief Clear all the error signals in the network.
         */
        void clearError();

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
         *
         * @note Use trainSequence() if the input data represents a time series.
         *       Only trainSequence() correctly backpropagates errors through time,
         *       resets the network between epochs and keeps sample in-order.
         */
        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   unsigned int batch_size,
                   unsigned int epochs);

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
                   unsigned int epochs);

        /**
         * @brief Train the network on a sequence of inputs and outputs
         *
         * @param inputs Matrix having one column per input sample
         * @param outputs Matrix having one column per output sample
         * @param epochs Number of epochs of training
         */
        void trainSequence(const Eigen::MatrixXf &inputs,
                           const Eigen::MatrixXf &outputs,
                           unsigned int epochs);

        /**
         * @brief Train the network on a sequence of inputs and outputs, using
         *        weights for the output neurons.
         *
         * @param weights Matrix having one column per weight vector. The weight
         *                vectors are used as described in trainSample().
         */
        void trainSequence(const Eigen::MatrixXf &inputs,
                           const Eigen::MatrixXf &outputs,
                           const Eigen::MatrixXf &weights,
                           unsigned int epochs);

    private:
        Float setExpectedOutput(const Vector &output, const Vector *weights);
        Float trainSample(const Vector &input, const Vector &output, const Vector *weights);
        void train(const Eigen::MatrixXf &inputs,
                   const Eigen::MatrixXf &outputs,
                   const Eigen::MatrixXf *weights,
                   unsigned int batch_size,
                   unsigned int epochs);
        void trainSequence(const Eigen::MatrixXf &inputs,
                           const Eigen::MatrixXf &outputs,
                           const Eigen::MatrixXf *weights,
                           unsigned int epochs);

    private:
        Port _input_port;
};

#endif
