#include "dense.h"

Dense::Dense(unsigned int outputs, Float learning_rate, Float decay)
: _input(nullptr),
  _learning_rate(learning_rate),
  _decay(decay)
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
    _bias = Vector::Random(outputs) * 0.01f;
    _d_bias = Vector::Zero(outputs);
    _avg_d_bias = Vector::Zero(outputs);

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
    _output.value.noalias() = _weights * _input->value + _bias;
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
        (_avg_d_weights.cwiseSqrt().array() + 1e-30).matrix()
    );
    _bias.noalias() -= (_learning_rate * _d_bias).cwiseQuotient(
        (_avg_d_bias.cwiseSqrt().array() + 1e-30).matrix()
    );
}

void Dense::clearError()
{
    _output.error.setZero();
    _d_weights.setZero();
    _d_bias.setZero();

    // Keep the moving averages as they are, so that they contain interesting
    // statistics about the general behavior of the gradients.
}
