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
        /**
         * @brief Backpropagation through time method used
         */
        enum BpttVariant {
            Standard,       /*!< @brief e(t) = backprop(y(t+1), e(t+1)), standard BPTT used by everyone */
            Experimental    /*!< @brief e(t) = (backprop(y(t+1), e(t+1)) + e(t+1))/length, experimental BPTT that gives better results in some cases */
        };

        static BpttVariant bptt_variant;

    public:
        AbstractRecurrentNetworkNode();
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
        unsigned int currentTimestep();

    protected:
        /**
         * @brief Copy the values of the recurrent nodes from time t to time t+1
         */
        void forwardRecurrent();

        /**
         * @brief Copy the error of the recurrent nodes from time t to time t-1
         */
        void backwardRecurrent();

    private:
        struct N
        {
            AbstractNode *node;
            std::vector<Port *> storage;
        };

        unsigned int _timestep;
        unsigned int _max_timestep;
        float _error_normalization;

        std::vector<N> _recurrent_nodes;
};

#endif
