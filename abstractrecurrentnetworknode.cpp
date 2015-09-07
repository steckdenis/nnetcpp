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

#include "abstractrecurrentnetworknode.h"

#include <assert.h>

AbstractRecurrentNetworkNode::BpttVariant AbstractRecurrentNetworkNode::bptt_variant = AbstractRecurrentNetworkNode::Standard;

AbstractRecurrentNetworkNode::AbstractRecurrentNetworkNode()
: _timestep(0)
{
}

AbstractRecurrentNetworkNode::~AbstractRecurrentNetworkNode()
{
    // Ensure that the storage is emptied
    reset();
}

void AbstractRecurrentNetworkNode::addRecurrentNode(AbstractNode *node)
{
    N n;

    n.node = node;

    _recurrent_nodes.push_back(n);
}

void AbstractRecurrentNetworkNode::forward()
{
    AbstractNetworkNode::forward();
    forwardRecurrent();
}

void AbstractRecurrentNetworkNode::forwardRecurrent()
{
    // Copy the value of the recurrent nodes to the storage
    for (N &n : _recurrent_nodes) {
        assert(n.storage.size() > _timestep);

        n.storage[_timestep]->value = n.node->output()->value;
    }
}

void AbstractRecurrentNetworkNode::backward()
{
    AbstractNetworkNode::backward();
    backwardRecurrent();
}

void AbstractRecurrentNetworkNode::backwardRecurrent()
{
    // Copy the error of the recurrent nodes in the storage at previous time step
    if (_timestep > 0) {
        for (N &n : _recurrent_nodes) {
            assert(n.storage.size() > _timestep);

            switch (bptt_variant) {
            case Standard:
                // Remove error[t] that was already present at the node output
                // before backprop started, and that therefore has to be removed
                // so that the node error contains only the error from the current
                // time step
                n.storage[_timestep - 1]->error = (n.node->output()->error - n.storage[_timestep]->error).cwiseMin(10.0f).cwiseMax(-10.0f);
                break;

            case Experimental:
                // The node error contains backprop(y(t), e(t)) + e(t), because
                // the error of this node is not cleared between time steps.
                // Divide by the sequence length in order to avoid an exponential
                // increase of the error over time.
                n.storage[_timestep - 1]->error = n.node->output()->error * _error_normalization;
                break;
            }
        }
    }
}

void AbstractRecurrentNetworkNode::reset()
{
    AbstractNetworkNode::reset();

    // Clear the storage
    for (N &n : _recurrent_nodes) {
        for (Port *port : n.storage) {
            delete port;
        }

        n.storage.clear();
    }

    // Reset the timestep counter, so that a next sequence can be shorter than
    // the one just finished.
    _max_timestep = 0;
}

void AbstractRecurrentNetworkNode::setCurrentTimestep(unsigned int timestep)
{
    // Let AbstractNetworkNode reset the error signals of all the nodes in the cell.
    AbstractNetworkNode::setCurrentTimestep(timestep);

    for (N &n : _recurrent_nodes) {
        assert(timestep <= n.storage.size());

        // Add a new port if needed
        if (timestep == n.storage.size()) {
            int size = n.node->output()->value.rows();

            n.storage.push_back(new Port);

            n.storage.back()->value = Vector::Zero(size);
            n.storage.back()->error = Vector::Zero(size);
        }

        if (timestep > 0) {
            // Set the value of the recurrent connection to the value at time t-1
            n.node->output()->value = n.storage[timestep - 1]->value;
        } else {
            // Reset the recurrent value
            n.node->output()->value.setZero();
        }

        // Set the error of the recurrent node to the error computed at time t
        n.node->output()->error = n.storage[timestep]->error;
    }

    // Use the timestep
    _timestep = timestep;

    // Keep track of the length of the sequence, this is used for normalizing
    // backpropagated errors
    _max_timestep = std::max(_max_timestep, timestep);
    _error_normalization = 1.0f / float(_max_timestep);
}

unsigned int AbstractRecurrentNetworkNode::currentTimestep()
{
    return _timestep;
}
