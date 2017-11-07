#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <functional>

namespace aica
{
	class Network
	{
	public:
		Network(int input, int hidden, int out, double learningrate);

		void Train(xt::xarray<float> inputs, xt::xarray<float> targets);

		xt::xarray<float> Query(xt::xarray<float> inputs);
	private:
		xt::xarray<float> m_wih;
		xt::xarray<float> m_who;
		int m_inputs;
		int m_hiddens;
		int m_outputs;
		double m_learningRate;
		std::function<double(double)> m_activation;
	};
}