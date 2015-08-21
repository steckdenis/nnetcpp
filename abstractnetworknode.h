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

#ifndef __ABSTRACTNETWORKNODE_H__
#define __ABSTRACTNETWORKNODE_H__

#include "abstractnode.h"

/**
 * @brief Node made of a network of sub-nodes (recurrent nodes for instance)
 */
class AbstractNetworkNode : public AbstractNode
{
    public:
        virtual ~AbstractNetworkNode();

        /**
         * @brief Add a node to this network. The first node receives the input,
         *        the last one produces the output of the network.
         */
        void addNode(AbstractNode *node);

        virtual void forward();
        virtual void backward();
        virtual void update();
        virtual void clearError();
        virtual void reset();

        virtual void setCurrentTimestep(unsigned int timestep);

    protected:
        std::vector<AbstractNode *> _nodes;
};

#endif
