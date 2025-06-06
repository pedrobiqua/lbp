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
            for (int i = 1; i < src.rows; ++i) {
                for (int j = 1; j < src.cols; ++j) {
                    // Faço essa transformação se não vem caracter ascii
                    uchar center = src.at<uchar>(i, j);
                    std::cout << static_cast<int>(src.at<uchar>(i, j)) << " ";
                    
                    uchar code = 0;
                    // kernel 3x3 | Fazer isso de forma mais inteligente, pois quero passar o kernel como parametro
                    code |= (src.at<uchar>(i-1, j-1) >= center) << 7; // Canto superior esquerdo
                    code |= (src.at<uchar>(i-1, j  ) >= center) << 6; // canto esquerdo
                    code |= (src.at<uchar>(i-1, j+1) >= center) << 5; // canto inferior esquerdo
                    code |= (src.at<uchar>(i,   j+1) >= center) << 4; // canto superior
                    code |= (src.at<uchar>(i+1, j+1) >= center) << 3; // canto superior direito
                    code |= (src.at<uchar>(i+1, j  ) >= center) << 2; // canto direito
                    code |= (src.at<uchar>(i+1, j-1) >= center) << 1; // canto inferior direito
                    code |= (src.at<uchar>(i,   j-1) >= center) << 0; // canto inferior

                    // Adicionar o code em uma estrutura do número binário montado
                }
                std::cout << std::endl;
            }
        }
    };
} // namespace lbp_library
