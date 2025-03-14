#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iostream>

namespace slam {
    namespace common {

        // -----------------------------------------------------------------------------
        /**
         * @brief Converts a basic data type (int, float, etc.) to a string.
         * @tparam T Type of input variable.
         * @param t Value to be converted.
         * @return String representation of 't'.
         */
        template <typename T>
        std::string toString(const T& t) {
            std::ostringstream ss;
            ss << t;
            return ss.str();
        }

        // -----------------------------------------------------------------------------
        /**
         * @brief Reads a delimited file into a matrix of strings.
         * @param file_name Path to the file.
         * @param delim Delimiter character (default: ',').
         * @return A vector of vectors containing parsed strings.
         * @throws std::invalid_argument If the file cannot be opened.
         */
        inline std::vector<std::vector<std::string>> loadData(const std::string_view file_name, char delim = ',') {
            std::ifstream in_file(file_name.data());
            if (!in_file.is_open()) {
                throw std::invalid_argument("Error opening input file: " + std::string(file_name));
            }

            std::vector<std::vector<std::string>> str_matrix;
            std::string line;

            while (std::getline(in_file, line)) {
                if (line.empty()) continue; // Skip empty lines
                
                std::vector<std::string> row;
                std::istringstream ss(line);
                std::string field;

                while (std::getline(ss, field, delim)) {
                    row.push_back(std::move(field));
                }

                if (!row.empty()) {
                    str_matrix.push_back(std::move(row));
                }
            }

            in_file.close(); // Ensure file is closed before returning
            return str_matrix;
        }

        // -----------------------------------------------------------------------------
        /**
         * @brief Writes a matrix of strings into a delimited file.
         * @param file_name Path to the file.
         * @param str_matrix Matrix of strings to be written.
         * @param delim Delimiter character (default: ',').
         * @throws std::invalid_argument If the input matrix is empty.
         * @throws std::runtime_error If the file cannot be opened.
         */
        inline void writeData(const std::string_view file_name, const std::vector<std::vector<std::string>>& str_matrix, char delim = ',') {
            if (str_matrix.empty()) {
                throw std::invalid_argument("Provided matrix of strings is empty.");
            }

            std::ofstream out_file(file_name.data());
            if (!out_file.is_open()) {
                throw std::runtime_error("Failed to open output file: " + std::string(file_name));
            }

            for (const auto& row : str_matrix) {
                if (row.empty()) continue; // Skip empty rows
                
                out_file << row.front();
                for (size_t i = 1; i < row.size(); ++i) {
                    out_file << delim << row[i];
                }
                out_file << '\n';
            }

            out_file.close(); // Ensure file is properly closed
        }

    } // namespace common
} // namespace slam
