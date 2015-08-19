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

#ifndef __ABSTRACTRECURRENTNETWORKNODE_H__
#define __ABSTRACTRECURRENTNETWORKNODE_H__

#include "abstractnetworknode.h"

#include <vector>

/**
 * @brief Network node that allows recurrent connections in it
 *
 * Recurrent connections need careful playing with setCurrentTimestep(), forward()
 * and backward(). This node provides implementations for these methods so that
 * managing recurrent connexions is easier.
 */
class AbstractRecurrentNetworkNode : public AbstractNetworkNode
{
    public:
        AbstractRecurrentNetworkNode(unsigned int size);
        virtual ~AbstractRecurrentNetworkNode();

        /**
         * @brief Add a recurrent node, that propagates errors and values between
         *        time steps.
         */
        void addRecurrentNode(AbstractNode *node);

        virtual void forward();
        virtual void backward();
        virtual void reset();

        virtual void setCurrentTimestep(unsigned int timestep);

    private:
        struct N
        {
            AbstractNode *node;
            std::vector<Port *> storage;
        };

        unsigned int _timestep;
        unsigned int _size;

        std::vector<N> _nodes;
};

#endif
