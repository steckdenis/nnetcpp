#ifndef __TANH_H__
#define __TANH_H__

#include "abstractnode.h"

/**
 * @brief Tanh activation layer
 */
class Tanh : public AbstractNode
{
    public:
        /**
         * @brief Apply the tanh activation function to the input, produce the
         *        same number of outputs.
         */
        Tanh();

        /**
         * @brief Set the input port of this node. The output will take the shape
         *        of the input.
         */
        void setInput(Port *input);

        virtual Port *output();
        virtual void forward();
        virtual void backward();
        virtual void update();
        virtual void clearError();

    private:
        Port *_input;
        Port _output;
};

#endif