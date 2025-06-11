#include <lbp/knn.hpp>
#include <lbp/lbp.hpp>

int main(int argc, char const *argv[])
{
    // Passos:
    //     1. Ler a pasta com as fotos dos carros
    cv::Mat src = cv::imread("/home/pedro/projects/lbp_pedro/examples/imgs/carro-esporte.png", cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        std::cerr << "Erro ao carregar a imagem. Certifique-se de que o caminho está correto!" << std::endl;
        return -1;
    }
    // Padronizo as imagens para o tamanho 32x32
    cv::resize(src, src, cv::Size(32, 32), 0, 0, cv::INTER_AREA);

    //     2. Aplicar o LBP nas imagens para criar o dataset de treino e teste
    lbp_library::LBP lbp;
    cv::Mat dst;
    lbp.compute(src, dst);
    std::vector<int> histogram = lbp.getHistogram();

    // Converto para essa estrutura que eu montei, onde tem os features e
    std::vector<knn_ml::data_point> train = knn_ml::convert_data_points(histogram, 0);

    //     2. Rodar o KNN para os dados de treino/teste
    int k = 3;
    knn_ml::knn my_knn("euclidean", k);
    my_knn.fit(train);
    // my_knn.predict();

    //     3. Mostrar os resultados

    return 0;
}
