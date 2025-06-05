#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat image = cv::imread("/home/pedro/projects/lbp_pedro/examples/imgs/carro-esporte.png", cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        std::cerr << "Erro ao carregar a imagem!" << std::endl;
        return -1;
    }

    int kernel_size = 3;

    for (int y = 0; y <= image.rows - kernel_size; ++y)
    {
        for (int x = 0; x <= image.cols - kernel_size; ++x)
        {
            // Extrai uma região da imagem (ROI: Region of Interest)
            cv::Mat roi = image(cv::Rect(x, y, kernel_size, kernel_size));

            // Calcula a média da região (como exemplo de operação do kernel)
            cv::Scalar mean_value = cv::mean(roi);

            std::cout << "Kernel @ (" << x << ", " << y << ") -> Média: " << mean_value[0] << std::endl;

            // Aqui você pode aplicar uma operação mais complexa com um kernel personalizado.
        }
    }

    return 0;
}