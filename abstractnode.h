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

#ifndef __ABSTRACTNODE_H__
#define __ABSTRACTNODE_H__

#include <vector>
#include <Eigen/Dense>

class NetworkSerializer;

// Typedefs so that using doubles instead of singles is easy
typedef Eigen::VectorXf Vector;
typedef Eigen::MatrixXf Matrix;
typedef float Float;

/**
 * @brief Node in a neural network
 *
 * A node take inputs and produces outputs, and is a generalization of whatever
 * can appear in a neural network. Dense connections, activation functions, memory
 * cells and error measures are nodes.
 *
 * For instance, a single-hidden-layer feed-forward neural network can be made
 * by connecting nodes in this way :
 *
 * input(Network) -> Dense -> Sigmoid -> Dense -> Sigmoid -> MSE
 *
 * The output of the last sigmoid being the output of the network, and MSE standing
 * for "mean square error", used to compute the error during the backpropagation
 * step.
 */
class AbstractNode
{
    public:
        /**
         * @brief Data input into a node
         *
         * A port has a value (what has been predicted or produced) and an error.
         * Nodes update the errors of their input ports, and produce values on
         * their output ports.
         */
        struct Port
        {
            Vector value;   /*!< @brief Value of this port, produced by its owner */
            Vector error;   /*!< @brief Error of this port, updated by its consumers */
        };

        AbstractNode() {}
        virtual ~AbstractNode() {}

        /**
         * @brief Serialize the weights of this node (if any)
         */
        virtual void serialize(NetworkSerializer &serializer) { (void) serializer; }

        /**
         * @brief Deserialize the weights of this node (if applicable)
         */
        virtual void deserialize(NetworkSerializer &serializer) { (void) serializer; }

        /**
         * @brief Output port of this dense network
         */
        virtual Port *output() = 0;

        /**
         * @brief Forward pass from the inputs to the outputs of this node
         */
        virtual void forward() = 0;

        /**
         * @brief Backward pass from the outputs to the inputs, updating the errors
         */
        virtual void backward() = 0;

        /**
         * @brief Update the parameters of this node based on the gradients
         *        computed by backward()
         */
        virtual void update() = 0;

        /**
         * @brief Clear the errors in this node, not touching the parameters or
         *        memory cells
         */
        virtual void clearError() = 0;

        /**
         * @brief Reset any memory stored in this network (but does not touch
         *        its parameters)
         */
        virtual void reset() {}

        /**
         * @brief If the input is a sequence, inform the node of a new position
         *        in the sequence.
         *
         * The time-step given can either be a new one (the 10 first inputs have
         * been seen, now is the 11th), or a time-step that was already visited
         * (the network is rewinded to the 9th time step so that error can be
         * backpropagated from the 10th to the 9th time step).
         *
         * The default implementation calls clearError() so that the errors are
         * not accumulated from one time step to the other.
         */
        virtual void setCurrentTimestep(unsigned int timestep)
        {
            (void) timestep;
            clearError();
        }
};

#endif
