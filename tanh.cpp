#include "tanh.h"

#include <cmath>

static Float _exp(Float x)
{
    // The constant 30 is also used by CLSTM
    if (x < -30) return exp(-30);
    if (x > 30) return exp(30);
    return exp(x);
}

static Float _expmx(Float x)
{
    return _exp(-x);
}

static Float _tanh(Float x)
{
    return 2.0f / (1.0f + _expmx(x)) - 1.0f;
}

static Float _tanh_d(Float x)
{
    Float expmx = _expmx(x);

    return (1.0f + expmx) * (1.0f + expmx) / (2.0f * expmx);
}

Tanh::Tanh()
: _input(nullptr)
{
}

void Tanh::setInput(Port *input)
{
    unsigned int inputs = input->value.rows();

    _output.value = Vector::Zero(inputs);
    _output.error = Vector::Zero(inputs);
    _input = input;
}

AbstractNode::Port *Tanh::output()
{
    return &_output;
}

void Tanh::forward()
{
    _output.value.noalias() = _input->value.unaryExpr(&_tanh);
}

void Tanh::backward()
{
    _input->error.noalias() += _output.error.cwiseProduct(
        (1.0f + _output.value.array().square()).matrix()
    );
}

void Tanh::clearError()
{
    _output.error.setZero();
}

void Tanh::update()
{
}
