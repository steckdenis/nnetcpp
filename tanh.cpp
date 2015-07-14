#include "tanh.h"

#include <cmath>

static Float _expmx(Float x)
{
    return std::exp(-x);
}

static Float _tanh(Float x)
{
    return 2.0f / (1.0f + _expmx(x)) - 1.0f;
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
    auto expmx = _output.value.unaryExpr(&_expmx).array();

    _input->error.noalias() += (
        _output.error.array() * (1.0f + expmx) * (1.0f + expmx) / (2.0f * expmx)
    ).matrix();
}

void Tanh::clearError()
{
}

void Tanh::update()
{
}
