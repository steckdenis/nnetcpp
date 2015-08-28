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

#include "abstractnetworknode.h"

AbstractNetworkNode::~AbstractNetworkNode()
{
    for (AbstractNode *node : _nodes) {
        delete node;
    }
}

void AbstractNetworkNode::addNode(AbstractNode *node)
{
    _nodes.push_back(node);
}

void AbstractNetworkNode::serialize(NetworkSerializer &serializer)
{
    // Serialize the nodes
    for (AbstractNode *node : _nodes) {
        node->serialize(serializer);
    }
}

void AbstractNetworkNode::deserialize(NetworkSerializer &serializer)
{
    // Deserialize the nodes
    for (AbstractNode *node : _nodes) {
        node->deserialize(serializer);
    }
}

void AbstractNetworkNode::forward()
{
    for (AbstractNode *node : _nodes) {
        node->forward();
    }
}

void AbstractNetworkNode::backward()
{
    for (int i=_nodes.size()-1; i>=0; --i) {
        AbstractNode *node = _nodes[i];

        node->backward();
    }
}

void AbstractNetworkNode::update()
{
    for (AbstractNode *node : _nodes) {
        node->update();
    }
}

void AbstractNetworkNode::clearError()
{
    for (AbstractNode *node : _nodes) {
        node->clearError();
    }
}

void AbstractNetworkNode::reset()
{
    AbstractNode::reset();

    for (AbstractNode *node : _nodes) {
        node->reset();
    }
}

void AbstractNetworkNode::setCurrentTimestep(unsigned int timestep)
{
    for (AbstractNode *node : _nodes) {
        node->setCurrentTimestep(timestep);
    }
}
