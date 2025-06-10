#include <lbp/knn.hpp>
#include <iostream>
#include <armadillo>
#include <vector>
#include "matplotlibcpp.h"  // Incluindo matplotlibcpp para visualização

namespace plt = matplotlibcpp;

using namespace knn_ml;

int main() {
    // Criando dados de treino
    // std::vector<data_point> train_data = {
    //     data_point({1.0, 2.0}, 0),
    //     data_point({1.5, 1.8}, 0),
    //     data_point({5.0, 8.0}, 1),
    //     data_point({6.0, 9.0}, 1),
    //     data_point({1.0, 0.6}, 0),
    //     data_point({9.0, 11.0}, 1)
    // };

    // // Criando dados de teste
    // std::vector<data_point> test_data = {
    //     data_point({1.2, 1.9}, -1), // label -1 significa que estamos testando (desconhecido)
    //     data_point({6.0, 10.0}, -1)
    // };

    // int k = 3;
    // // Instanciando o KNN
    // knn my_knn("euclidean", k); // você pode trocar para "manhattan" ou "minkowski"
    // my_knn.fit(train_data);

    // // Fazendo previsões
    // std::vector<int> predictions = my_knn.predict(test_data);

    // // Imprimindo os resultados
    // std::cout << "Predições:\n";
    // for (size_t i = 0; i < test_data.size(); ++i) {
    //     std::cout << "Ponto de teste " << i + 1 << ": Classe prevista = " << predictions[i] << std::endl;
    // }

    // // Preparando os dados para visualização
    // std::vector<double> train_x, train_y, test_x, test_y;
    // std::vector<int> train_labels;

    // // Adicionando pontos de treino
    // for (const auto& point : train_data) {
    //     train_x.push_back(point.features[0]); // Coordenada x
    //     train_y.push_back(point.features[1]); // Coordenada y
    //     train_labels.push_back(point.label);   // Rótulo
    // }

    // // Adicionando pontos de teste
    // for (size_t i = 0; i < test_data.size(); ++i) {
    //     test_x.push_back(test_data[i].features[0]); // Coordenada x
    //     test_y.push_back(test_data[i].features[1]); // Coordenada y
    // }

    // // Criando o gráfico
    // plt::figure_size(800, 600);

    // // // Plotando os pontos de treino
    // // for (size_t i = 0; i < train_data.size(); ++i) {
    // //     if (train_labels[i] == 0) {
    // //         plt::scatter({train_x[i]}, {train_y[i]}, 50, {{"color", "blue"}});
    // //     } else {
    // //         plt::scatter({train_x[i]}, {train_y[i]}, 50, {{"color", "red"}});
    // //     }
    // // }

    // // // Plotando os pontos de teste
    // // for (size_t i = 0; i < test_data.size(); ++i) {
    // //     if (predictions[i] == 0) {
    // //         plt::scatter({test_x[i]}, {test_y[i]}, 100, {{"color", "cyan"}, {"label", "Teste classe 0"}});
    // //     } else {
    // //         plt::scatter({test_x[i]}, {test_y[i]}, 100, {{"color", "green"}, {"label", "Teste classe 1"}});
    // //     }
    // // }

    // // Personalizando o gráfico
    // plt::title("KNN: Classificação de pontos");
    // plt::xlabel("Feature 1");
    // plt::ylabel("Feature 2");
    // plt::legend();
    
    // // Exibindo o gráfico
    // plt::show();

    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 1, 3, 5};

    plt::scatter(x, y);
    plt::save("teste_scatter.png");

    return 0;
}
