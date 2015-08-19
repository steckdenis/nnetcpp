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

AbstractRecurrentNetworkNode::AbstractRecurrentNetworkNode(unsigned int size)
: _timestep(0),
  _size(size)
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

    _nodes.push_back(n);
}

void AbstractRecurrentNetworkNode::forward()
{
    AbstractNetworkNode::forward();

    // Copy the value of the recurrent nodes to the storage
    assert(_storage.size() > _timestep);

    for (N &n : _nodes) {
        n.storage[_timestep]->value = n.node->output()->value;
    }
}

void AbstractRecurrentNetworkNode::backward()
{
    AbstractNetworkNode::backward();

    // Copy the error of the recurrent nodes in the storage at previous time step
    assert(_storage.size() > _timestep);

    if (_timestep > 0) {
        for (N &n : _nodes) {
            n.storage[_timestep - 1]->error = n.node->output()->error;
        }
    }
}

void AbstractRecurrentNetworkNode::reset()
{
    AbstractNode::reset();

    // Clear the storage
    for (N &n : _nodes) {
        for (Port *port : n.storage) {
            delete port;
        }

        n.storage.clear();
    }
}

void AbstractRecurrentNetworkNode::setCurrentTimestep(unsigned int timestep)
{
    assert(timestep <= _storage.size());

    // Let AbstractNetworkNode reset the error signals of all the nodes in the cell.
    AbstractNetworkNode::setCurrentTimestep(timestep);

    for (N &n : _nodes) {
        // Add a new port if needed
        if (timestep == n.storage.size()) {
            n.storage.push_back(new Port);

            n.storage.back()->value = Vector::Zero(_size);
            n.storage.back()->error = Vector::Zero(_size);
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
}
