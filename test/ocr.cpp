#include <ann.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <mnist/mnist_reader.hpp>
#include <chrono>

int main(int argc,char** argv)
{
	aica::Network net(784,100,10,0.2);

	auto start = std::chrono::high_resolution_clock::now();
	auto dataset = mnist::read_dataset();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Loaded MNIST dataset after " << duration.count() << "ms" << std::endl;
	start = end;

	int idx = 0;
	for (auto img : dataset.training_images)
	{
		auto lbl = dataset.training_labels[idx];
		
		xt::xarray<uint8_t> uimg;
		std::array<std::size_t,1> shape = { 768 };
		auto arr = xt::xadapt(img, shape, xt::layout_type::row_major);
		auto ftensor = xt::cast<float>(arr);
		auto normalized = ftensor/255.0*0.99+0.01;
	
		xt::xtensor<float,1> targets = xt::zeros<float>({ 10 }) + 0.01;
		targets[lbl] = 0.99;

		net.Train(normalized, targets);
		++idx;
	}

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Finished training after " << duration.count() << "ms" << std::endl;

	idx = 0;
	for (auto img : dataset.test_images)
	{
		auto lbl = dataset.test_labels[idx];

		xt::xarray<uint8_t> uimg;
		std::array<std::size_t, 1> shape = { 768 };
		auto arr = xt::xadapt(img, shape, xt::layout_type::row_major);
		auto ftensor = xt::cast<float>(arr);
		auto normalized = ftensor / 255.0*0.99 + 0.01;

		auto result = net.Query(normalized);

		std::cout << "Correct: " << lbl;
		for (int i = 0; i < 10; ++i)
		{
			
			std::cout << result[i] << std::endl;
		}
		++idx;
	}

    return 0;
}