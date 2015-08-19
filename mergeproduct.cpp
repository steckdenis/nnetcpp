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

#include "mergeproduct.h"

MergeProduct::MergeProduct()
{
}

void MergeProduct::forward()
{
    _output.value.setOnes();

    // Add the inputs together
    for (Port *input : _inputs) {
        _output.value.array() *= input->value.array();
    }
}

void MergeProduct::backward()
{
    // If f(a, b, c) = a * b * c, df/da = b * c = f(a, b, c) / a. So, divide the
    // output by one of the inputs in order to compute the gradient of the product
    // with regard to this input, and multiply the output error by this gradient
    // in order to produce the input error.
    for (Port *input : _inputs) {
        input->error.noalias() += _output.error.cwiseProduct(
            _output.value.cwiseQuotient((input->value.array() + 1e-20).matrix())
        );
    }
}
