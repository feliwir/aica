#include <ann.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <mnist/mnist_reader.hpp>
#include <chrono>

#define MNIST_SIZE 784

int main(int argc,char** argv)
{
	aica::Network net(MNIST_SIZE,100,10,0.2);

	auto start = std::chrono::high_resolution_clock::now();
	auto dataset = mnist::read_dataset();
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Loaded MNIST dataset after " << duration.count() << "ms" << std::endl;
	start = end;

	int idx = 0;
	for (const auto& img : dataset.training_images)
	{
		auto& lbl = dataset.training_labels[idx];
		
		xt::xarray<uint8_t> uimg;
		std::array<std::size_t,2> shape = { MNIST_SIZE,1 };
		auto arr = xt::xadapt(img, shape, xt::layout_type::row_major);
		auto ftensor = xt::cast<float>(arr);
		auto normalized = ftensor/255.0*0.99+0.01;
	
		xt::xtensor<float,2> targets = xt::zeros<float>({ 10,1 }) + 0.01;
		targets(lbl,0) = 0.99;

		net.Train(normalized, targets);
		++idx;
	}

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Finished training after " << duration.count() << "ms" << std::endl;
	start = end;

	std::vector<int> correct;

	idx = 0;
	for (const auto& img : dataset.test_images)
	{
		uint8_t lbl = dataset.test_labels[idx];

		xt::xarray<uint8_t> uimg;
		std::array<std::size_t, 2> shape = { MNIST_SIZE,1 };
		auto arr = xt::xadapt(img, shape, xt::layout_type::row_major);
		auto ftensor = xt::cast<float>(arr);
		auto normalized = ftensor / 255.0*0.99 + 0.01;

		auto result = net.Query(normalized);

		int max = 0;
		for (int i = 0; i < 10; ++i)		
			if (result(i, 0) > result(max, 0))
				max = i;
		
		if (max == lbl)
			correct.push_back(1);
		else
			correct.push_back(0);
		++idx;
	}

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Finished testing after " << duration.count() << "ms" << std::endl;

	int sum = std::accumulate(correct.begin(), correct.end(), 0);
	double acc = (double)sum / correct.size();

	std::cout << "Recognition accuracy is: " << std::setprecision(5)  << acc << std::endl;

	int n = 0;
	std::cin >> n;
    return 0;
}