#include "ann.hpp"
#include "activations.hpp"
#include <xtensor/xrandom.hpp>
#include <xtensor/xvectorize.hpp>
#include <xtensor-blas/xlinalg.hpp>

aica::Network::Network(int inputs, int hiddens, int outs, double lr) :
	m_inputs(inputs), m_hiddens(hiddens), m_outputs(outs), m_learningRate(lr)
{
	m_wih = xt::random::randn({m_hiddens,m_inputs }, 0.0, std::pow(m_inputs , -0.5));
	m_who = xt::random::randn({m_outputs,m_hiddens}, 0.0, std::pow(m_hiddens, -0.5));

	m_activation = Activations::Sigmoid<float>;
}

void aica::Network::Train(xt::xarray<float> inputs, xt::xarray<float> targets)
{
	xt::xarray<float> hiddenInputs = xt::linalg::dot(m_wih, inputs);
	xt::xarray<float> hiddenOutputs = m_activation(hiddenInputs);

	xt::xarray<float> finalInputs = xt::linalg::dot(m_who, hiddenOutputs);
	xt::xarray<float> finalOutputs = m_activation(finalInputs);

	xt::xarray<float> outputErrors = targets - finalOutputs;

	xt::xarray<float> tmp_t = xt::transpose(m_who);
	xt::xarray<float> hiddenErrors = xt::linalg::dot(tmp_t, outputErrors);

	tmp_t = xt::transpose(hiddenOutputs);
	m_who += m_learningRate * xt::linalg::dot((outputErrors * finalOutputs * (1.0f - finalOutputs)),
		tmp_t);

	tmp_t = xt::transpose(inputs);
	m_wih += m_learningRate * xt::linalg::dot((hiddenErrors * hiddenOutputs * (1.0f - hiddenOutputs)),
		tmp_t);
}

xt::xarray<float> aica::Network::Query(xt::xarray<float> inputs)
{
	xt::xarray<float> hiddenInputs = xt::linalg::dot(m_wih, inputs);
	xt::xarray<float> hiddenOutputs = m_activation(hiddenInputs);

	xt::xarray<float> finalInputs = xt::linalg::dot(m_who, hiddenOutputs);
	xt::xarray<float> finalOutputs = m_activation(finalInputs);

	return finalOutputs;
}

//void aica::Network::Train(xt::xarray<float> inputs, xt::xarray<float> targets)
//{
//	auto& activation = xt::vectorize(m_activation);
//
//	auto hiddenInputs = xt::linalg::dot(inputs,m_wih);
//	auto hiddenOutputs = activation(hiddenInputs);
//
//	auto finalInputs = xt::linalg::dot(hiddenOutputs,m_who);	
//}

