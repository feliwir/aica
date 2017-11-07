#include <ann.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <mnist/mnist_reader.hpp>

int main(int argc,char** argv)
{
	aica::Network net(784,100,10,0.2);
	auto dataset = mnist::read_dataset();

	int idx = 0;
	for (auto img : dataset.training_images)
	{
		auto lbl = dataset.training_labels[idx];
		
		xt::xarray<uint8_t> uimg;
		std::array<std::size_t,1> shape = { 768 };
		auto arr = xt::xadapt(img, shape, xt::layout_type::row_major);
		auto ftensor = xt::cast<float>(arr);
		auto normalized = ftensor/255.0*0.99+0.01;
		auto targets = xt::xtensor<float, 1>::from_shape({ 10 });
		
		for (int i = 0; i < 10; ++i)
		{
			if (i == lbl)
				targets[i] = 0.99;
			else
				targets[i] = 0.01;
		}

		net.Train(normalized, targets);
		++idx;
	}

	for (auto img : dataset.test_images)
	{
		auto lbl = dataset.test_labels[idx];

		xt::xarray<uint8_t> uimg;
		std::array<std::size_t, 1> shape = { 768 };
		auto arr = xt::xadapt(img, shape, xt::layout_type::row_major);
		auto ftensor = xt::cast<float>(arr);
		auto normalized = ftensor / 255.0*0.99 + 0.01;

		auto result = net.Query(normalized);

		for (int i = 0; i < 10; ++i)
		{
			std::cout << result[i] << std::endl;
		}
		++idx;
	}

    return 0;
}