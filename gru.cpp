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

#include "gru.h"
#include "activation.h"
#include "dense.h"
#include "mergeproduct.h"
#include "mergesum.h"

GRU::GRU(unsigned int size, Float learning_rate, Float decay)
{
    // Intantiate all the nodes used by a GRU cell
    MergeSum *inputs = new MergeSum;
    TanhActivation *input_activation = new TanhActivation;

    MergeSum *updates = new MergeSum;
    SigmoidActivation *update_activation = new SigmoidActivation;
    OneMinusActivation *oneminus_update_activation = new OneMinusActivation;
    MergeProduct *update_times_output = new MergeProduct;
    MergeProduct *oneminus_update_times_input = new MergeProduct;
    MergeSum *output = new MergeSum;                                           // z*_output + (1-z)*_inputs

    MergeSum *resets = new MergeSum;
    SigmoidActivation *reset_activation = new SigmoidActivation;
    MergeProduct *reset_times_output = new MergeProduct;                       // wired back to _inputs through a Dense

    Dense *loop_output_to_updates = new Dense(size, learning_rate, decay);
    Dense *loop_output_to_resets = new Dense(size, learning_rate, decay);
    Dense *loop_reset_times_output_to_inputs = new Dense(size, learning_rate, decay);

    // Wire-up everything, taking care that only outputs with an already-known
    // size are connected to inputs.
    resets->addInput(loop_output_to_resets->output());
    updates->addInput(loop_output_to_updates->output());
    inputs->addInput(loop_reset_times_output_to_inputs->output());

    input_activation->setInput(inputs->output());
    update_activation->setInput(updates->output());
    oneminus_update_activation->setInput(update_activation->output());

    update_times_output->addInput(update_activation->output());
    update_times_output->addInput(output->output());
    oneminus_update_times_input->addInput(input_activation->output());
    oneminus_update_times_input->addInput(oneminus_update_activation->output());

    output->addInput(update_times_output->output());
    output->addInput(oneminus_update_times_input->output());

    reset_activation->setInput(resets->output());
    reset_times_output->addInput(reset_activation->output());
    reset_times_output->addInput(output->output());

    loop_output_to_resets->setInput(output->output());
    loop_output_to_updates->setInput(output->output());
    loop_reset_times_output_to_inputs->setInput(reset_times_output->output());

    // Put everything in a list, in the order in which the forward pass will be run
    addNode(loop_output_to_updates);
    addNode(loop_output_to_resets);

    addNode(resets);
    addNode(reset_activation);
    addNode(reset_times_output);

    addNode(loop_reset_times_output_to_inputs);

    addNode(inputs);
    addNode(input_activation);

    addNode(updates);
    addNode(update_activation);
    addNode(oneminus_update_activation);
    addNode(update_times_output);
    addNode(oneminus_update_times_input);

    addNode(output);

    // Ensure that h(0) = 0
    _inputs = inputs;
    _resets = resets;
    _updates = updates;
    _output = output;

    reset();
}

AbstractNode::Port *GRU::output()
{
    return _output->output();
}

void GRU::addInput(Port *input)
{
    _inputs->addInput(input);
}

void GRU::addR(Port *r)
{
    _resets->addInput(r);
}

void GRU::addZ(Port *z)
{
    _updates->addInput(z);
}

void GRU::reset()
{
    AbstractNode::reset();

    // Set the output (and its error) to zero
    _output->output()->value.setZero();
    _output->output()->error.setZero();
}
