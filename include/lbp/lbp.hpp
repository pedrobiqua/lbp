#include <string.h>
#include <opencv2/core.hpp>

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

        void compute(const cv::Mat &src, cv::Mat &dst) const
        {
        }
    };
} // namespace lbp_library
