#include <opencv2/opencv.hpp>
#include <lbp/lbp.hpp>
#include "matplotlibcpp.h" // Certifique-se de que está no caminho de include

namespace plt = matplotlibcpp;

int main(int argc, char const *argv[])
{
    cv::Mat src = cv::imread("/home/pedro/projects/lbp_pedro/examples/imgs/carro-esporte.png", cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        std::cerr << "Erro ao carregar a imagem. Certifique-se de que o caminho está correto!" << std::endl;
        return -1;
    }

    lbp_library::LBP lbp;
    cv::Mat dst;
    std::cout << lbp.getDefaultName() << std::endl;
    lbp.compute(src, dst);

    std::vector<int> result = lbp.getResult(); // std::vector<float> ou std::vector<double>
    std::vector<int> histogram = lbp.getHistogram();

    // PLOT DO HISTOGRAMA
    std::vector<int> bins(histogram.size());
    std::iota(bins.begin(), bins.end(), 0); // Gera [0, 1, 2, ..., N-1]

    plt::figure_size(800, 400);
    plt::bar(bins, histogram);
    plt::title("Histograma LBP");
    plt::xlabel("Padrões LBP");
    plt::ylabel("Frequência");
    plt::show();

    // Mostrar a imagem transformada
    // cv::imshow("Imagem LBP", dst);
    // int k = cv::waitKey(0);
    return 0;
}
