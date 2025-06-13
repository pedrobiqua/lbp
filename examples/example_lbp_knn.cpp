#include <lbp/knn.hpp>
#include <lbp/lbp.hpp>

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

int main()
{
    std::vector<knn_ml::data_point> dataset;
    lbp_library::LBP lbp;

    const std::string base_path = "/home/pedro/projects/lbp_pedro/examples/imgs";
    const int width = 32, height = 32;

    // Função para carregar as imagens de uma pasta
    auto load_images = [&](const std::string &folder, int label) {
        for (const auto &entry : fs::directory_iterator(folder)) {
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Erro ao carregar: " << entry.path() << std::endl;
                continue;
            }

            cv::resize(img, img, cv::Size(width, height));
            cv::Mat dst;
            lbp.compute(img, dst);
            std::vector<int> hist = lbp.getHistogram();

            auto data = knn_ml::convert_data_points(hist, label);
            dataset.insert(dataset.end(), data.begin(), data.end());
        }
    };

    // Carrega imagens das duas classes
    load_images(base_path + "2012-09-11/empty", 0);
    load_images(base_path + "2012-09-11/occupied", 1);

    if (dataset.empty()) {
        std::cerr << "Nenhum dado carregado!" << std::endl;
        return -1;
    }

    // Embaralha e separa treino/teste
    std::shuffle(dataset.begin(), dataset.end(), std::mt19937{std::random_device{}()});
    size_t train_size = static_cast<size_t>(dataset.size() * 0.7);

    std::vector<knn_ml::data_point> train(dataset.begin(), dataset.begin() + train_size);
    std::vector<knn_ml::data_point> test(dataset.begin() + train_size, dataset.end());

    std::cout << "Treinando com " << train.size() << " amostras, testando com " << test.size() << " amostras\n";

    // Treina o KNN
    knn_ml::knn knn_model("euclidean", 3);
    knn_model.fit(train);

    // Avalia no teste
    // int acertos = 0;
    // for (const auto &ponto : test) {
    //     int pred = knn_model.predict(ponto.features);
    //     if (pred == ponto.label)
    //         acertos++;
    // }

    // double acc = 100.0 * acertos / test.size();
    // std::cout << "Acurácia no conjunto de teste: " << acc << "%\n";

    // return 0;
}
