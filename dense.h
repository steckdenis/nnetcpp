#ifndef __DENSE_H__
#define __DENSE_H__

#include "abstractnode.h"

/**
 * @brief Dense fully-connected layer, with no activation function (linear activation)
 */
class Dense : public AbstractNode
{
    public:
        /**
         * @brief Make a dense connection between an input and the output of this node
         */
        Dense(unsigned int outputs, Float learning_rate, Float decay = 0.9f);

        /**
         * @brief Set the input port of this node
         */
        void setInput(Port *input);

        virtual Port *output();
        virtual void forward();
        virtual void backward();
        virtual void update();
        virtual void clearError();

    private:
        Port *_input;
        Float _learning_rate;
        Float _decay;

        Port _output;

        Matrix _weights;
        Matrix _d_weights;
        Matrix _avg_d_weights;
        Vector _bias;
        Vector _d_bias;
        Vector _avg_d_bias;
};

#endif