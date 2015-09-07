#include "cwrnn.h"
#include "dense.h"
#include "mergesum.h"

#include <iostream>
#include <assert.h>

CWRNN::CWRNN(unsigned int num_units, unsigned int size, Float learning_rate, Float decay)
: _learning_rate(learning_rate),
  _decay(decay)
{
    // Output node that sums the outputs of all the units
    _output = new MergeSum;

    // Create the units
    _units.resize(num_units);
    _unit_size = size / num_units;

    assert(_unit_size * num_units == size);

    for (unsigned int i=0; i<num_units; ++i) {
        Unit &unit = _units[i];

        unit.sum = new MergeSum;
        unit.activation = new TanhActivation;
        unit.skip = new LinearActivation;
        unit.output = new MergeSum;

        // Recurrent connections between the previous units and this one (this one included)
        for (unsigned int j=0; j<=i; ++j) {
            Dense *dense = new Dense(_unit_size, _learning_rate, _decay);
            Unit &prev_unit = _units[j];

            // HACK: this dense has the same number of inputs and outputs. First
            //       set itself as input so that it discovers its size, then
            //       set the proper input
            dense->setInput(dense->output());

            unit.inputs.push_back(dense);
            unit.sum->addInput(dense->output());
            addNode(dense);

            if (j == 0) {
                // Now that sum knows its size (thanks to the Dense nodes), other connections
                // can be made
                unit.activation->setInput(unit.sum->output());
                unit.output->addInput(unit.activation->output());
                unit.output->addInput(unit.skip->output());
                unit.skip->setInput(unit.output->output());
            }

            // HACK (contd.): set the real output of dense
            dense->setInput(prev_unit.output->output());
        }

        // The output of this unit is a recurrent node that has to be registered
        // as such
        addRecurrentNode(unit.output);
        addNode(unit.sum);
        addNode(unit.activation);
        addNode(unit.skip);
        addNode(unit.output);

        // Add the output of this unit to the general output
        _output->addInput(unit.output->output());
    }

    reset();
}

void CWRNN::addInput(Port *input)
{
    // Add the input to all the units, inserting a Dense between the input and
    // the units.
    for (Unit &unit : _units) {
        Dense *dense = new Dense(_unit_size, _learning_rate, _decay);

        dense->setInput(input);

        unit.sum->addInput(dense->output());
        unit.inputs.push_back(dense);
        _inputs.push_back(dense);

        addNode(dense);
    }
}

AbstractNode::Port *CWRNN::output()
{
    return _output->output();
}

template<typename EnabledFunc, typename DisabledFunc>
void CWRNN::forUnits(unsigned int t, EnabledFunc enabled, DisabledFunc disabled)
{
    // The first unit is enabled one timestep every 2^(num_units-1), the second
    // unit every 2^(num_units-2) timesteps, etc. The last unit is always enabled
    for (unsigned int i=0; i<_units.size(); ++i) {
        unsigned int period = 1 << (_units.size() - i - 1);

        if (t % period == 0) {
            enabled(_units[i]);
        } else {
            disabled(_units[i]);
        }
    }
}

void CWRNN::forward()
{
    // Propagate the input (or skip links) of the units
    forUnits(
        currentTimestep(),
        [](Unit &enabled) {
            for (Dense *dense : enabled.inputs) {
                dense->forward();
            }

            enabled.sum->forward();
            enabled.activation->forward();      // output = activation(sum(inputs)), skip is set to zero by Activation::clearError, called by setCurrentTimestep.
        },
        [](Unit &disabled) {
            disabled.skip->forward();           // output = output(t-1), activation is set to zero by Activation::clearError.
        }
    );

    // Forward the output of all the units (now that the last outputs are not
    // needed anymore)
    for (Unit &unit : _units) {
        unit.output->forward();
    }

    // Forward pass of the output node
    _output->forward();

    // Store the output of the recurrent nodes for later use
    AbstractRecurrentNetworkNode::forwardRecurrent();
}

void CWRNN::backward()
{
    // Backward pass of the output node
    _output->backward();

    // Propagate the error of the units to their inputs or skip links
    forUnits(
        currentTimestep(),
        [](Unit &enabled) {
            enabled.output->backward();         // Also set the error at the skip link, but this link will not be backpropagated
            enabled.activation->backward();
            enabled.sum->backward();

            for (Dense *dense : enabled.inputs) {
                dense->backward();
            }
        },
        [](Unit &disabled) {
            disabled.output->backward();        // Also set the error at "activation", but this error will not be backpropagated
            disabled.skip->backward();
        }
    );

    // Store the backpropagated error for later use
    AbstractRecurrentNetworkNode::backwardRecurrent();
}
