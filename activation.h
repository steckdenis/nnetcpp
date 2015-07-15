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
    if (x < -30) return exp(-30);
    if (x > 30) return exp(30);
    return exp(x);
}

struct Tanh
{
    Float operator()(Float x) const
    {
        return 2.0f / (1.0f + _exp(-x)) - 1.0f;
    }
};

struct dTanh
{
    Float operator()(Float y) const
    {
        return (1.0f + y*y);
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

}

// Instantiate the Activation templates
typedef Activation<nnetcppinternal::Tanh, nnetcppinternal::dTanh> TanhActivation;
typedef Activation<nnetcppinternal::Sigmoid, nnetcppinternal::dSigmoid> SigmoidActivation;

#endif