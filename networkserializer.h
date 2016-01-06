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

#ifndef __NETWORKSERIALIZER_H__
#define __NETWORKSERIALIZER_H__

#include <ostream>
#include <istream>
#include <vector>

/**
 * @brief Data store to/from which the weights of a neural network can be stored/retrieved
 */
class NetworkSerializer
{
    public:
        NetworkSerializer();

        /**
         * @brief Write a value to the buffer
         */
        void writeWeight(float value);

        /**
         * @brief Read a value from the buffer and advance its read pointer
         */
        float readWeight();

        /**
         * @brief Save the contents of the serializer to a file
         */
        void save(std::ostream &s);

        /**
         * @brief Load the contents of the serializer from a file
         */
        void load(std::istream &s);

        /**
         * @brief Pointer to the data currently in the serializer
         */
        float *data();

        /**
         * @brief Number of elements in the serializer
         */
        unsigned int size() const;

    private:
        std::vector<float> _data;
        unsigned int _pos;
};

#endif
