// #include <opencv2/opencv.hpp>
#include <iostream>
#include <lbp/lbp.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char const *argv[])
{
    std::cout << "Oie" << std::endl;

    cv::Mat image = cv::imread("/home/pedro/projects/lbp_pedro/examples/imgs/carro-esporte.png", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "Erro ao carregar a imagem. Certifique-se de que o caminho está correto!" << std::endl;
    }

    lbp_library::lbp lbp;
    std::cout << lbp.getDefaultName() << std::endl;
    lbp.compute(image);

    imshow("Display window", image);
    int k = cv::waitKey(0);
    return 0;
}
