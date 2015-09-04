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
         * @brief Value by which the gradient is multiplied between updates, a
         *        non-zero value allows the gradient to have "inertia" in its
         *        main direction.
         */
        static float momentum;

    public:
        /**
         * @brief Make a dense connection between an input and the output of this node
         */
        Dense(unsigned int outputs, Float learning_rate, Float decay = 0.9f, bool bias_initialized_at_one = false);

        /**
         * @brief Set the input port of this node
         */
        void setInput(Port *input);

        virtual void serialize(NetworkSerializer &serializer);
        virtual void deserialize(NetworkSerializer &serializer);

        virtual Port *output();
        virtual void forward();
        virtual void backward();
        virtual void update();
        virtual void clearError();
        virtual void reset();

        virtual void setCurrentTimestep(unsigned int timestep);

    private:
        Port *_input;
        Float _learning_rate;
        Float _decay;
        bool _bias_initialized_at_one;

        Port _output;

        Matrix _weights;
        Matrix _d_weights;
        Matrix _avg_d_weights;
        Vector _bias;
        Vector _d_bias;
        Vector _avg_d_bias;

        unsigned int _max_timestep;
};

#endif
