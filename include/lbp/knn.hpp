#ifndef KNN_H
#define KNN_H

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <iostream>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Namespace para KNN
namespace knn_ml {

// Estrutura de dados para pontos
struct data_point {
    arma::rowvec features; // Características do ponto
    int label; // Rótulo (classe) do ponto

    data_point(arma::rowvec f, int l)
        : features(f)
        , label(l)
    {
    }
};

class knn {
private:
    std::vector<data_point> train_data;
    std::string distance_metric;
    int k;

public:
    knn(const std::string& metric = "euclidean", const int& k = 3)
        : distance_metric(metric), k(k) {}

    ~knn() {
        // Aqui você pode colocar o código para liberar recursos, se necessário.
        // No caso dessa classe, parece que não há alocação dinâmica explícita, então o destruidor pode ser vazio.
    }

    void fit(const std::vector<data_point>& train) { train_data = train; }

    std::vector<int> predict(const std::vector<data_point>& test_data)
    {
        std::vector<int> predictions;

        // para cada ponto de teste, predizer a classe
        for (const auto& test_point : test_data) {
            int predicted_class = classify(test_point, k);
            predictions.push_back(predicted_class);
        }

        return predictions;
    }

private:
    double euclidean_distance(const arma::rowvec& a, const arma::rowvec& b)
    {
        return arma::norm(a - b, 2);
    }

    double manhattan_distance(const arma::rowvec& a, const arma::rowvec& b)
    {
        return arma::norm(a - b, 1); // Norm 1 (distância Manhattan)
    }

    double minkowski_distance(const arma::rowvec& a, const arma::rowvec& b, double p)
    {
        return std::pow(arma::norm(a - b, p), 1.0 / p); // Distância Minkowski
    }

    // Função auxiliar para calcular a distância
    double calculate_distance(const arma::rowvec& a, const arma::rowvec& b)
    {
        if (distance_metric == "euclidean") {
            return euclidean_distance(a, b);
        } else if (distance_metric == "manhattan") {
            return manhattan_distance(a, b);
        } else if (distance_metric == "minkowski") {
            double p = 3;
            return minkowski_distance(a, b, p);
        } else {
            throw std::invalid_argument("Métrica de distância desconhecida");
        }
    }

    // Função para classificação
    int classify(const data_point& test_point, int k)
    {
        std::vector<std::pair<double, int>> distances;

        // Calcular a distância de cada ponto de treino para o ponto de teste
        for (const auto& train_point : train_data) {
            double dist = calculate_distance(test_point.features, train_point.features);
            distances.push_back(std::make_pair(dist, train_point.label));
        }

        // Ordenar as distâncias
        std::sort(distances.begin(), distances.end());

        // Contar as classes dos k vizinhos mais próximos
        std::unordered_map<int, int> class_count;
        for (int i = 0; i < k; i++) {
            class_count[distances[i].second]++;
        }

        // Encontrar a classe com maior contagem
        int predicted_class = -1;
        int max_count = 0;
        for (const auto& entry : class_count) {
            if (entry.second > max_count) {
                max_count = entry.second;
                predicted_class = entry.first;
            }
        }

        return predicted_class;
    }
};

} // namespace knn_ml

#endif // KNN_H
