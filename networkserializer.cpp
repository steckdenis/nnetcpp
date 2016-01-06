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

#include "networkserializer.h"

#include <assert.h>

NetworkSerializer::NetworkSerializer()
: _pos(0)
{
}

void NetworkSerializer::writeWeight(float value)
{
    _data.push_back(value);
}

float NetworkSerializer::readWeight()
{
    assert(_pos < _data.size());

    return _data[_pos++];
}

void NetworkSerializer::save(std::ostream &s)
{
    s.write((const char *)_data.data(), _data.size() * sizeof(float));
}

void NetworkSerializer::load(std::istream &s)
{
    float v;

    while (!s.eof()) {
        s.read((char *)&v, sizeof(float));

        writeWeight(v);
    }
}

float *NetworkSerializer::data()
{
    return _data.data();
}

unsigned int NetworkSerializer::size() const
{
    return _data.size();
}
