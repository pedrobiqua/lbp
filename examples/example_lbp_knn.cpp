#include <lbp/knn.hpp>
#include <lbp/lbp.hpp>

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <random>
#include <algorithm>

#include <chrono>
#include "matplotlibcpp.h" // Incluindo matplotlibcpp para visualização

namespace fs = std::filesystem;
namespace plt = matplotlibcpp;

// Função que carrega a imagem, converte para 32x32, e usa a função lbp e guarda o resultado no formato data_point, para auxiliar na utilização do lbp
void load_images(const std::string &folder, int label,
                 std::vector<knn_ml::data_point> &dataset,
                 lbp_library::LBP &lbp,
                 int width = 32, int height = 32)
{
    for (const auto &entry : std::filesystem::directory_iterator(folder))
    {
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            std::cerr << "Erro ao carregar: " << entry.path() << std::endl;
            continue;
        }

        cv::resize(img, img, cv::Size(width, height));
        cv::Mat dst;
        lbp.compute(img, dst);
        std::vector<int> hist = lbp.getHistogram();

        dataset.emplace_back(hist, label);
    }
}

void save_confusion_matrix_to_csv(const std::vector<int> &predictions,
                                  const std::vector<knn_ml::data_point> &test)
{
    int confusion[2][2] = {{0, 0}, {0, 0}};
    for (size_t i = 0; i < test.size(); ++i)
    {
        int true_label = test[i].getLabel();
        int pred_label = predictions[i];
        confusion[true_label][pred_label]++;
    }

    std::ofstream file("confusion_matrix.csv");
    if (file.is_open())
    {
        file << "Predito\\Verdadeiro,Empty,Occupied\n";
        file << "Empty," << confusion[0][0] << "," << confusion[0][1] << "\n";
        file << "Occupied," << confusion[1][0] << "," << confusion[1][1] << "\n";
        file.close();
    }
    else
    {
        std::cerr << "Erro ao abrir o arquivo CSV!" << std::endl;
    }
}

int main()
{
    std::vector<knn_ml::data_point> dataset;
    lbp_library::LBP lbp;

    const std::string base_path = "/home/pedro/projects/lbp_pedro/examples/imgs/";
    const int width = 32, height = 32;
    std::cout << "Carregando os dados da pasta: " << base_path << std::endl;
    auto inicio_carregamento = std::chrono::high_resolution_clock::now();

    // Carrega imagens das duas classes
    load_images(base_path + "2012-09-11/Empty", 0, dataset, lbp);
    load_images(base_path + "2012-09-11/Occupied", 1, dataset, lbp);

    auto fim_carregamento = std::chrono::high_resolution_clock::now();
    auto tempo_carregamento = fim_carregamento - inicio_carregamento;
    long long segundos_carregamento = std::chrono::duration_cast<std::chrono::seconds>(tempo_carregamento).count();
    std::cout << "Tempo de carregamento: " << segundos_carregamento << std::endl;

    if (dataset.empty())
    {
        std::cerr << "Nenhum dado carregado!" << std::endl;
        return -1;
    }

    // Uso a função shuffle para embaralhar o meu array
    std::mt19937 rng(42); // ← Seed fixa para reprodutibilidade
    std::shuffle(dataset.begin(), dataset.end(), rng);
    size_t train_size = static_cast<size_t>(dataset.size() * 0.7);

    // Faço a separação de treino e teste, onde 70% são usados para treino e 30% para teste
    std::vector<knn_ml::data_point> train(dataset.begin(), dataset.begin() + train_size);
    std::vector<knn_ml::data_point> test(dataset.begin() + train_size, dataset.end());

    std::cout << "Treinando com " << train.size() << " amostras, testando com " << test.size() << " amostras" << std::endl;

    // Treina o KNN, basicamente passar a estrutura já montada
    knn_ml::knn knn_model("euclidean", 3);
    knn_model.fit(train);

    // Avalia no teste
    auto inicio = std::chrono::high_resolution_clock::now();
    std::vector<int> predictions = knn_model.predict(test);
    int acertos = 0;

    for (size_t i = 0; i < test.size(); ++i)
    {
        if (predictions[i] == test[i].getLabel())
            acertos++;
    }
    auto fim = std::chrono::high_resolution_clock::now();
    auto tempo = (fim - inicio);
    long long segundos = std::chrono::duration_cast<std::chrono::seconds>(tempo).count();

    double acc = 100.0 * acertos / test.size();
    std::cout << "Acurácia no conjunto de teste: " << acc << std::endl;
    std::cout << "Tempo das predições: " << segundos << std::endl;

    std::cout << "Plot dos resultados" << std::endl;
    save_confusion_matrix_to_csv(predictions, test);

    return 0;
}
