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

#include "dense.h"

Dense::Dense(unsigned int outputs, Float learning_rate, Float decay, bool bias_initialized_at_one)
: _input(nullptr),
  _learning_rate(learning_rate),
  _decay(decay),
  _bias_initialized_at_one(bias_initialized_at_one)
{
    // Prepare the output port
    _output.error.resize(outputs);
    _output.value.resize(outputs);
}

void Dense::setInput(Port *input)
{
    _input = input;

    // Initialize the weights and bias
    unsigned int inputs = _input->value.rows();
    unsigned int outputs = _output.value.rows();

    _weights = Matrix::Random(outputs, inputs) * 0.01f;
    _d_weights = Matrix::Zero(outputs, inputs);
    _avg_d_weights = Matrix::Zero(outputs, inputs);
    _d_bias = Vector::Zero(outputs);
    _avg_d_bias = Vector::Zero(outputs);

    if (_bias_initialized_at_one) {
        _bias = Vector::Ones(outputs);
    } else {
        _bias = Vector::Random(outputs) * 0.01f;
    }

    // Clear the error, so that the error is initialized for the first backpropagation
    // step.
    clearError();
}

AbstractNode::Port *Dense::output()
{
    return &_output;
}

void Dense::forward()
{
    _output.value.noalias() = _weights * _input->value;
    _output.value += _bias;
}

void Dense::backward()
{
    // Multiply the output errors by the weights to obtain the input errors
    _input->error.noalias() += _weights.transpose() * _output.error;

    // Update the gradient of the input parameters and biases
    _d_weights.noalias() -= _output.error * _input->value.transpose();
    _d_bias.noalias() -= _output.error;
}

void Dense::update()
{
    // Keep a moving average of the gradients
    _avg_d_weights = _decay * _avg_d_weights + (1.0f - _decay) * _d_weights.cwiseProduct(_d_weights);
    _avg_d_bias = _decay * _avg_d_bias + (1.0f - _decay) * _d_bias.cwiseProduct(_d_bias);

    // Perform the update using RMSprop
    _weights.noalias() -= (_learning_rate * _d_weights).cwiseQuotient(
        (_avg_d_weights.cwiseSqrt().array() + 1e-3).matrix()
    );
    _bias.noalias() -= (_learning_rate * _d_bias).cwiseQuotient(
        (_avg_d_bias.cwiseSqrt().array() + 1e-3).matrix()
    );
}

void Dense::clearError()
{
    _output.error.setZero();
    _d_weights *= 0.1f;
    _d_bias *= 0.1f;

    // Keep the moving averages as they are, so that they contain interesting
    // statistics about the general behavior of the gradients.
}

void Dense::setCurrentTimestep(unsigned int timestep)
{
    (void) timestep;

    // Clear the error signal but not the gradients
    _output.error.setZero();
}
