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

	m_activation = Activations::Sigmoid<double>;
}

void aica::Network::Train(xt::xarray<float> inputs, xt::xarray<float> targets)
{
	auto& activation = xt::vectorize(m_activation);

	auto hiddenInputs = xt::linalg::dot(inputs, m_wih);
	auto hiddenOutputs = activation(hiddenInputs);

	auto finalInputs = xt::linalg::dot(hiddenOutputs, m_who);
	auto finalOutputs = activation(finalInputs);

	auto outputErrors = targets - finalOutputs;

	auto hiddenErrors = xt::linalg::dot(xt::transpose(m_who), outputErrors);

	m_who += m_learningRate * xt::linalg::dot((outputErrors * finalOutputs * (1.0 - finalOutputs)),
											xt::transpose(hiddenOutputs));

	m_wih += m_learningRate * xt::linalg::dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)),
		xt::transpose(inputs));
}

xt::xarray<float> aica::Network::Query(xt::xarray<float> inputs)
{
	auto& activation = xt::vectorize(m_activation);

	auto hiddenInputs = xt::linalg::dot(inputs, m_wih);
	auto hiddenOutputs = activation(hiddenInputs);

	auto finalInputs = xt::linalg::dot(hiddenOutputs, m_who);
	auto finalOutputs = activation(finalInputs);

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

