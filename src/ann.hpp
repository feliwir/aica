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

		void Train(const xt::xarray<float>& inputs,const xt::xarray<float>& targets);

		xt::xarray<float> Query(const xt::xarray<float>& inputs) const;
	private:
		xt::xarray<float> m_wih;
		xt::xarray<float> m_who;
		int m_inputs;
		int m_hiddens;
		int m_outputs;
		float m_learningRate;
		std::function<xt::xarray<float>(xt::xarray<float>)> m_activation;
	};
}