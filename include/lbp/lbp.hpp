#include <string.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/core/mat.hpp>

namespace lbp_library
{
    /**
     * Aqui estou herdando o Algorithm para poder utilizar algumas funções do OpenCV
     */
    class LBP : public cv::Algorithm
    {
    private:
        /* data */
        std::vector<int> lbpResult_; // Não é exatamente um histograma e mais o resultado do lbp

    public:
        LBP() {}
        ~LBP() {}

        cv::String getDefaultName() const override
        {
            return "lbp_pedro.MyLBP";
        }

        void compute(const cv::Mat &src, cv::Mat &dst)
        {
            // verifica se vem vazio
            CV_Assert(!src.empty() && src.channels() == 1);
            // Estou pegando o canal 1, de escala de cinza e não RGB
            // std::cout << "Linhas: " << src.rows << " Colunas " << src.cols << std::endl; // DEBUG
            dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

            for (int i = 1; i < src.rows; ++i)
            {
                for (int j = 1; j < src.cols; ++j)
                {
                    // Faço essa transformação se não vem caracter ascii
                    uchar center = src.at<uchar>(i, j);
                    // std::cout << static_cast<int>(src.at<uchar>(i, j)) << " "; // DEBUG

                    // MOVIMENTAÇÃO DO "KERNEL"
                    uchar code = 0;
                    code |= (src.at<uchar>(i - 1, j - 1) >= center) << 7; // Canto superior esquerdo
                    code |= (src.at<uchar>(i - 1, j) >= center) << 6;     // canto esquerdo
                    code |= (src.at<uchar>(i - 1, j + 1) >= center) << 5; // canto inferior esquerdo
                    code |= (src.at<uchar>(i, j + 1) >= center) << 4;     // canto superior
                    code |= (src.at<uchar>(i + 1, j + 1) >= center) << 3; // canto superior direito
                    code |= (src.at<uchar>(i + 1, j) >= center) << 2;     // canto direito
                    code |= (src.at<uchar>(i + 1, j - 1) >= center) << 1; // canto inferior direito
                    code |= (src.at<uchar>(i, j - 1) >= center) << 0;     // canto inferior

                    // Adicionar o code em uma estrutura do número binário montado
                    this->lbpResult_.push_back(static_cast<int>(code));
                    dst.at<uchar>(i, j) = code;
                }
                // std::cout << std::endl;
            }
        }

        std::vector<int> getResult() const
        {
            return this->lbpResult_;
        }

        std::vector<int> getHistogram() const
        {
            std::vector<int> histograma(256, 0);
            for (auto &&i : this->lbpResult_)
            {
                histograma[i]++;
            }

            return histograma;
        }
    };
} // namespace lbp_library
