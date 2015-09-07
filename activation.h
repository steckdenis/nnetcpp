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

#ifndef __ACTIVATION_H__
#define __ACTIVATION_H__

#include "abstractnode.h"

/**
 * @brief Template class for activation layers
 */
template<typename F, typename DF>
class Activation : public AbstractNode
{
    public:
        /**
         * @brief Apply the activation function to the input, produce the
         *        same number of outputs.
         */
        Activation() {}

        /**
         * @brief Set the input port of this node. The output will take the shape
         *        of the input.
         */
        void setInput(Port *input)
        {
            unsigned int inputs = input->value.rows();

            _output.value = Vector::Zero(inputs);
            _output.error = Vector::Zero(inputs);
            _input = input;
        }

        virtual Port *output()
        {
            return &_output;
        }

        virtual void forward()
        {
            _output.value.noalias() = _input->value.unaryExpr<F>();
        }

        virtual void backward()
        {
            _input->error.noalias() += _output.error.cwiseProduct(
                _output.value.unaryExpr<DF>()
            );
        }

        virtual void clearError()
        {
            _output.error.setZero();
            _output.value.setZero();
        }

        virtual void update()
        {
        }

    private:
        Port *_input;
        Port _output;
};

namespace nnetcppinternal
{

inline Float _exp(Float x)
{
    // The constant 30 is also used by CLSTM
    if (x < -30.0f) return std::exp(-30.0f);
    if (x > 30.0f) return std::exp(30.0f);
    return std::exp(x);
}

struct Tanh
{
    Float operator()(Float x) const
    {
        return 2.0f / (1.0f + _exp(-2.0f * x)) - 1.0f;
    }
};

struct dTanh
{
    Float operator()(Float y) const
    {
        return 1.0f - y*y;
    }
};

struct Sigmoid
{
    Float operator()(Float x) const
    {
        return 1.0f / (1.0f + _exp(-x));
    }
};

struct dSigmoid
{
    Float operator()(Float y) const
    {
        return y * (1.0f - y);
    }
};

struct OneMinus
{
    Float operator()(Float x) const
    {
        return 1.0f - x;
    }
};

struct dOneMinus
{
    Float operator()(Float y) const
    {
        (void) y;
        return -1.0f;
    }
};

struct Linear
{
    Float operator()(Float x) const
    {
        return x;
    }
};

struct dLinear
{
    Float operator()(Float y) const
    {
        (void) y;
        return 1.0f;
    }
};

}

// Instantiate the Activation templates
/**
 * @brief Tanh (output from -1 to 1) activation
 */
typedef Activation<nnetcppinternal::Tanh, nnetcppinternal::dTanh> TanhActivation;

/**
 * @brief Sigmoid (output from 0 to 1) activation
 */
typedef Activation<nnetcppinternal::Sigmoid, nnetcppinternal::dSigmoid> SigmoidActivation;

/**
 * @brief Output = 1 - input
 */
typedef Activation<nnetcppinternal::OneMinus, nnetcppinternal::dOneMinus> OneMinusActivation;

/**
 * @brief Output = input
 */
typedef Activation<nnetcppinternal::Linear, nnetcppinternal::dLinear> LinearActivation;

#endif
