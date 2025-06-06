#include <string.h>
#include <opencv2/core.hpp>
#include <iostream>

namespace lbp_library
{
    /**
     * Aqui estou herdando o Algorithm para poder utilizar algumas funções do OpenCV
     */
    class lbp : public cv::Algorithm
    {
    private:
        /* data */
        std::string nome_;

    public:
        /*
        Testando a herança
        */
        cv::String getDefaultName() const override
        {
            return "lbp_pedro.MyLBP";
        }

        void compute(const cv::Mat &src) const
        {
            // TODO: Montar o kernel e fazer o histograma com o valor do kernel, depois vou querer plotar ele
            // Estou pegando o canal 1, de escala de cinza e não RGB
            for (int i = 0; i < src.rows; ++i) {
                for (int j = 0; j < src.cols; ++j) {
                    // Faço essa transformação se não vem caracter ascii
                    std::cout << static_cast<int>(src.at<uchar>(i, j)) << " ";
                }
                std::cout << std::endl;
            }
        }
    };
} // namespace lbp_library
