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
         */
        void setExpectedOutput(const Vector &output);

        /**
         * @brief Set the error signals at the output of this network and
         *        back-propagate it, without performing any gradient update
         */
        void setError(const Vector &error);

        /**
         * @brief Perform one gradient update using the error computed by the
         *        last calls to setExpectedOutput() and setError()
         *
         * @note The error signal is zeroed-out after a call to this function
         */
        void update();

        /**
         * @brief Shortcut method that performs one gradient update on a sample
         */
        void trainSample(const Vector &input, const Vector &output);

    private:
        AbstractNode::Port _input_port;

        std::vector<AbstractNode *> _nodes;
};

#endif