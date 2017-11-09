#pragma once
#include <cmath>

namespace aica
{
    class Activations
    {
    public:


        template<class T>
        static inline xt::xarray<T> Sigmoid(xt::xarray<T>  x)
        {
			return (T)1.0 / ((T)1.0 + xt::exp(-x));
        }; 
    };
}