#ifndef __ABSTRACTNODE_H__
#define __ABSTRACTNODE_H__

#include <vector>
#include <Eigen/Dense>

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
};

#endif