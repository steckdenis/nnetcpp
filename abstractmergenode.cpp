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

#include "abstractmergenode.h"

AbstractMergeNode::AbstractMergeNode()
{
}

AbstractNode::Port *AbstractMergeNode::output()
{
    return &_output;
}

void AbstractMergeNode::addInput(Port *input)
{
    unsigned int dim = input->value.rows();

    if (_inputs.size() == 0) {
        // First input to be added, its size defines the shape of the output
        _output.value = Vector::Zero(dim);
        _output.error = Vector::Zero(dim);
    }

    _inputs.push_back(input);
}

void AbstractMergeNode::clearError()
{
    _output.error.setZero();
    _output.value.setZero();
}

void AbstractMergeNode::update()
{
}
