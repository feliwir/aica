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
			return 1 / (1 + exp(-x));
        }; 
    };
}