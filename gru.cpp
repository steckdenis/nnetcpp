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
    _nodes.push_back(loop_output_to_updates);
    _nodes.push_back(loop_output_to_resets);

    _nodes.push_back(resets);
    _nodes.push_back(reset_activation);
    _nodes.push_back(reset_times_output);

    _nodes.push_back(loop_reset_times_output_to_inputs);

    _nodes.push_back(inputs);
    _nodes.push_back(input_activation);

    _nodes.push_back(updates);
    _nodes.push_back(update_activation);
    _nodes.push_back(oneminus_update_activation);
    _nodes.push_back(update_times_output);
    _nodes.push_back(oneminus_update_times_input);

    _nodes.push_back(output);

    // Ensure that h(0) = 0
    _inputs = inputs;
    _resets = resets;
    _updates = updates;
    _output = output;

    reset();
}

GRU::~GRU()
{
    for (AbstractNode *node : _nodes) {
        delete node;
    }
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

void GRU::forward()
{
    for (AbstractNode *node : _nodes) {
        node->forward();
    }
}

void GRU::backward()
{
    for (int i=_nodes.size()-1; i>=0; --i) {
        _nodes[i]->backward();
    }
}

void GRU::clearError()
{
    for (AbstractNode *node : _nodes) {
        node->clearError();
    }
}

void GRU::update()
{
    for (AbstractNode *node : _nodes) {
        node->update();
    }
}

void GRU::reset()
{
    AbstractNode::reset();

    // Set the output (and its error) to zero
    _output->output()->value.setZero();
    _output->output()->error.setZero();
}
