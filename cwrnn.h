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

#ifndef __CWRNN_H__
#define __CWRNN_H__

#include "abstractrecurrentnetworknode.h"
#include "activation.h"

#include <vector>

class Dense;
class MergeSum;

/**
 * @brief Clockwork RNN
 *
 * Implementation based on the description of "A Clockwork RNN", Koutnìk,
 * Greff, Gomez and Schmidhuber, 2014, arXiv:1v1153.2041.
 */
class CWRNN : public AbstractRecurrentNetworkNode
{
    public:
        /**
         * @brief Layer of Clockwork RNN units. All the input and output ports of this
         *        layer have the same shape.
         *
         * @param num_units Number of units, each unit i having a time resolution
         *        of 2^i. This number must divide @p size.
         */
        CWRNN(unsigned int num_units,
              unsigned int size,
              Float learning_rate,
              Float decay = 0.9f);

        /**
         * @brief Add an X input to this network
         *
         * @note The input does not need to be the output port of a Dense since
         *       CWRNN automatically adds Dense nodes between its input and the
         *       Clockwork units. For instance, you can simply pass Network::inputPort()
         *       as a parameter to this method.
         */
        void addInput(Port *input);

        virtual Port *output();
        virtual void forward();
        virtual void backward();

    private:
        /**
         * @brief Iterate over the units and call a functor depending on whether
         *        they are enabled or disabled
         */
        template<typename EnabledFunc, typename DisabledFunc>
        void forUnits(unsigned int t, EnabledFunc enabled, DisabledFunc disabled);

    private:
        struct Unit {
            std::vector<Dense *> inputs;
            MergeSum *sum;
            TanhActivation *activation;

            LinearActivation *skip;

            MergeSum *output;
        };

        std::vector<Unit> _units;
        std::vector<Dense *> _inputs;
        MergeSum *_output;

        unsigned int _unit_size;
        float _learning_rate;
        float _decay;
};

#endif
