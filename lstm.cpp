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

#include "lstm.h"
#include "activation.h"
#include "dense.h"
#include "mergeproduct.h"
#include "mergesum.h"

LSTM::LSTM(unsigned int size, Float learning_rate, Float decay)
{
    // Intantiate all the nodes used by a LSTM cell
    MergeSum *inputs = new MergeSum;
    TanhActivation *input_activation = new TanhActivation;

    MergeSum *input_gate = new MergeSum;
    SigmoidActivation *input_gate_activation = new SigmoidActivation;

    MergeSum *forget_gate = new MergeSum;
    SigmoidActivation *forget_gate_activation = new SigmoidActivation;

    MergeSum *output_gate = new MergeSum;
    SigmoidActivation *output_gate_activation = new SigmoidActivation;

    MergeProduct *input_times_input_gate = new MergeProduct;
    MergeProduct *cells_times_forget_gate = new MergeProduct;
    MergeSum *cells = new MergeSum;
    TanhActivation *cells_activation = new TanhActivation;
    MergeProduct *cells_times_output_gate = new MergeProduct;

    Dense *loop_output_to_output_gate = new Dense(size, learning_rate, decay);
    Dense *loop_output_to_input_gate = new Dense(size, learning_rate, decay);
    Dense *loop_output_to_forget_gate = new Dense(size, learning_rate, decay);
    Dense *loop_output_to_input = new Dense(size, learning_rate, decay);

    // Wire-up everything, taking care that only outputs with an already-known
    // size are connected to inputs.
    inputs->addInput(loop_output_to_input->output());
    input_gate->addInput(loop_output_to_input_gate->output());
    forget_gate->addInput(loop_output_to_forget_gate->output());
    output_gate->addInput(loop_output_to_forget_gate->output());

    input_activation->setInput(inputs->output());
    input_gate_activation->setInput(input_gate->output());
    forget_gate_activation->setInput(forget_gate->output());
    output_gate_activation->setInput(output_gate->output());

    input_times_input_gate->addInput(input_gate_activation->output());
    input_times_input_gate->addInput(input_activation->output());

    cells_times_forget_gate->addInput(forget_gate_activation->output());
    cells_times_forget_gate->addInput(cells->output());

    cells->addInput(input_times_input_gate->output());
    cells->addInput(cells_times_forget_gate->output());
    cells_activation->setInput(cells->output());

    cells_times_output_gate->addInput(output_gate_activation->output());
    cells_times_output_gate->addInput(cells_activation->output());

    loop_output_to_forget_gate->setInput(cells_times_output_gate->output());
    loop_output_to_input_gate->setInput(cells_times_output_gate->output());
    loop_output_to_output_gate->setInput(cells_times_output_gate->output());
    loop_output_to_input->setInput(cells_times_output_gate->output());

    // Put everything in a list, in the order in which the forward pass will be run
    _nodes.push_back(loop_output_to_forget_gate);
    _nodes.push_back(loop_output_to_input);
    _nodes.push_back(loop_output_to_input_gate);
    _nodes.push_back(loop_output_to_output_gate);

    _nodes.push_back(inputs);
    _nodes.push_back(input_activation);
    _nodes.push_back(input_gate);
    _nodes.push_back(input_gate_activation);
    _nodes.push_back(forget_gate);
    _nodes.push_back(forget_gate_activation);
    _nodes.push_back(output_gate);
    _nodes.push_back(output_gate_activation);

    _nodes.push_back(input_times_input_gate);
    _nodes.push_back(cells_times_forget_gate);
    _nodes.push_back(cells);
    _nodes.push_back(cells_activation);
    _nodes.push_back(cells_times_output_gate);

    // Ensure that h(0) = 0
    _inputs = inputs;
    _ingates = input_gate;
    _outgates = output_gate;
    _forgetgates = forget_gate;
    _cells = cells;
    _output = cells_times_output_gate;

    reset();
}

LSTM::~LSTM()
{
    for (AbstractNode *node : _nodes) {
        delete node;
    }
}

AbstractNode::Port *LSTM::output()
{
    return _output->output();
}

void LSTM::addInput(Port *input)
{
    _inputs->addInput(input);
}

void LSTM::addInGate(Port *in)
{
    _ingates->addInput(in);
}

void LSTM::addOutGate(Port *out)
{
    _outgates->addInput(out);
}

void LSTM::addForgetGate(Port *forget)
{
    _forgetgates->addInput(forget);
}

void LSTM::forward()
{
    for (AbstractNode *node : _nodes) {
        node->forward();
    }
}

void LSTM::backward()
{
    for (int i=_nodes.size()-1; i>=0; --i) {
        _nodes[i]->backward();
    }
}

void LSTM::clearError()
{
    for (AbstractNode *node : _nodes) {
        node->clearError();
    }
}

void LSTM::update()
{
    for (AbstractNode *node : _nodes) {
        node->update();
    }
}

void LSTM::reset()
{
    AbstractNode::reset();

    // Clear the content of the memory cells
    _cells->output()->value.setZero();
    _cells->output()->error.setZero();

    // Clear the output, so that the different loops from output to ???gate
    // will not propagate invalid data
    _output->output()->value.setZero();
    _output->output()->error.setZero();
}
