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
#include "dense.h"
#include "mergeproduct.h"
#include "mergesum.h"

#include <assert.h>

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
    LinearActivation *real_output = new LinearActivation;                      // "usable" output, receives error from the expected output of the unit, nothing from t+1
    LinearActivation *recurrent_output = new LinearActivation;                 // "recurrent" output, receives error from t+1

    MergeSum *resets = new MergeSum;
    SigmoidActivation *reset_activation = new SigmoidActivation;
    MergeProduct *reset_times_output = new MergeProduct;                       // wired back to _inputs through a Dense

    Dense *loop_output_to_updates = new Dense(size, learning_rate, decay, true);    // Bias updates to 1, so that the cell starts by being "transparent" (letting information flow through it)
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
    update_times_output->addInput(recurrent_output->output());                  // Z*output uses the recurrent connection and will contribute error to it.
    oneminus_update_times_input->addInput(input_activation->output());
    oneminus_update_times_input->addInput(oneminus_update_activation->output());

    output->addInput(update_times_output->output());
    output->addInput(oneminus_update_times_input->output());
    real_output->setInput(output->output());
    recurrent_output->setInput(output->output());

    reset_activation->setInput(resets->output());
    reset_times_output->addInput(reset_activation->output());
    reset_times_output->addInput(real_output->output());                        // reset*output uses the real output, so they will not contribute errors to recurrent_output

    loop_output_to_resets->setInput(real_output->output());                     // The loops from output to Z and R use real_output, so no error will be backpropagated to t-1
    loop_output_to_updates->setInput(real_output->output());
    loop_reset_times_output_to_inputs->setInput(reset_times_output->output());

    // Put everything in a list, in the order in which the forward pass will be run
    addNode(loop_output_to_updates);            // setCurrentTimeStep has properly set the output of real_output and recurrent_output, so these loops can be used.
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
    addNode(recurrent_output);                   // This line and the next one allow the cell to fully reach time step t, and allow error from t+1 to flow back in the entire cell
    addNode(real_output);

    // Register the recurrent output as a recurrent node
    addRecurrentNode(recurrent_output);

    // Ensure that h(0) = 0
    _inputs = inputs;
    _resets = resets;
    _updates = updates;
    _real_output = real_output;
    _recurrent_output = recurrent_output;

    reset();
}

AbstractNode::Port *GRU::output()
{
    return _real_output->output();
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

void GRU::setCurrentTimestep(unsigned int timestep)
{
    // Handle _recurrent_output (set with setRecurrentNode)
    AbstractRecurrentNetworkNode::setCurrentTimestep(timestep);

    // Ensure that _real_output has the same value than _recurrent_output, because
    // it is also used at some places. Keep its error to zero, because it will
    // receive it from the outside world and not from any recurrent connection
    _real_output->output()->value = _recurrent_output->output()->value;
}
