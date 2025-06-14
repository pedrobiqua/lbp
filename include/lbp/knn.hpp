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

namespace knn_ml
{

    class data_point
    {
    private:
        arma::rowvec features_;
        int label_;

    public:
        data_point(const arma::rowvec &f, int l)
            : features_(f), label_(l) {}

        data_point(const std::vector<int> &f, int l)
            : label_(l)
        {
            std::vector<double> temp(f.begin(), f.end());
            this->features_ = arma::rowvec(temp);
        }

        const arma::rowvec &getFeatures() const
        {
            return this->features_;
        }

        const int &getLabel() const
        {
            return this->label_;
        }
    };

    class knn
    {
    private:
        std::vector<data_point> train_data;
        std::string distance_metric;
        int k;

    public:
        knn(const std::string &metric = "euclidean", const int &k = 3)
            : distance_metric(metric), k(k) {}

        ~knn() {}

        void fit(const std::vector<data_point> &train)
        {
            train_data = train;
        }

        std::vector<int> predict(const std::vector<data_point> &test_data)
        {
            std::vector<int> predictions;
            for (const auto &test_point : test_data)
            {
                int predicted_class = classify(test_point, k);
                predictions.push_back(predicted_class);
            }
            return predictions;
        }

    private:
        double euclidean_distance(const arma::rowvec &a, const arma::rowvec &b)
        {
            // Distância Euclidiana
            return arma::accu(arma::square(a - b));
        }

        double manhattan_distance(const arma::rowvec &a, const arma::rowvec &b)
        {
            return arma::norm(a - b, 1);
        }

        double minkowski_distance(const arma::rowvec &a, const arma::rowvec &b, double p)
        {
            return std::pow(arma::accu(arma::pow(arma::abs(a - b), p)), 1.0 / p);
        }

        double calculate_distance(const arma::rowvec &a, const arma::rowvec &b)
        {
            if (distance_metric == "euclidean")
                return euclidean_distance(a, b);
            if (distance_metric == "manhattan")
                return manhattan_distance(a, b);
            if (distance_metric == "minkowski")
                return minkowski_distance(a, b, 3);
            throw std::invalid_argument("Métrica de distância desconhecida");
        }

        int classify(const data_point &test_point, int k)
        {
            std::vector<std::pair<double, int>> distances;
            for (const auto &train_point : train_data)
            {
                double dist = calculate_distance(test_point.getFeatures(), train_point.getFeatures());
                distances.emplace_back(dist, train_point.getLabel());
            }

            // Ordeno as distâncias calculadas, para facilitar na parte dos votos
            std::sort(distances.begin(), distances.end());

            std::unordered_map<int, int> class_count;
            for (int i = 0; i < k; ++i)
            {
                class_count[distances[i].second]++;
            }

            int predicted_class = -1, max_count = 0;
            for (const auto &entry : class_count)
            {
                if (entry.second > max_count)
                {
                    max_count = entry.second;
                    predicted_class = entry.first;
                }
            }

            return predicted_class;
        }
    };

} // namespace knn_ml

#endif // KNN_H
