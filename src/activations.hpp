#pragma once
#include <cmath>

namespace aica
{
    class Activations
    {
    public:
        template<class T>
        static inline T Sigmoid(T x)
        {
			return (T)1.0 / ((T)1.0 + exp(-x));
        }; 
    };
}