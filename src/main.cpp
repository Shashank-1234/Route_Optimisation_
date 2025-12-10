#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

namespace py = pybind11;

// Helper to read CSV into 2D vector
std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::vector<std::vector<double>> matrix;
    
    // Try opening file directly
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Try looking in parent directory (if running from build/)
        std::string parent_path = "../" + filename;
        file.open(parent_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename + " or " + parent_path);
        }
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    return matrix;
}

int main() {
    std::cout << "[C++ Host] Starting Logistics Engine..." << std::endl;
    
    try {
        // 1. Initialize Embedded Python Interpreter
        py::scoped_interpreter guard{};
        
        // 2. Add current directory to Python path so we can import our plugin
        py::module_ sys = py::module_::import("sys");
        sys.attr("path").attr("append")(".");
        
        // 3. Import the optimization plugin
        std::cout << "[C++ Host] Loading Python Optimization Plugin..." << std::endl;
        py::module_ plugin = py::module_::import("optimization_plugin");
        
        // 4. Load Data (In C++)
        std::cout << "[C++ Host] Reading Matrix Data (CSV)..." << std::endl;
        auto dist_matrix = read_csv("data/distance.csv");
        auto time_matrix = read_csv("data/time.csv");
        
        std::cout << "[C++ Host] Data Loaded. Rows: " << dist_matrix.size() << std::endl;
        
        // 5. Call Python Function
        // We pass C++ std::vector directly; pybind11 handles conversion to Python lists
        std::cout << "[C++ Host] Handing off to Python Solver..." << std::endl;
        py::object result = plugin.attr("solve_mission")(dist_matrix, time_matrix);
        
        // 6. Process Results (In C++)
        // Check for status first
        if (result.contains("status") && result["status"].cast<std::string>() == "success") {
            std::cout << "[C++ Host] Optimization Success. Extracting metrics..." << std::endl;
            
            // Extract total emissions safely
            double total_emissions = 0.0;
            if (result.contains("total_emissions")) {
                total_emissions = result["total_emissions"].cast<double>();
            }
            
            std::cout << "[C++ Host] Extracting routes..." << std::endl;
            py::list py_routes = result["routes"].cast<py::list>();
            
            std::cout << "\n==========================================" << std::endl;
            std::cout << "   MISSION SUCCESSFUL (Report by C++)" << std::endl;
            std::cout << "==========================================" << std::endl;
            std::cout << "Total Network Emissions: " << total_emissions << " g CO2" << std::endl;
            std::cout << "Active Fleet Size: " << py_routes.size() << " Vehicles" << std::endl;
            std::cout << "------------------------------------------" << std::endl;
            
            int v_idx = 1;
            for (auto item : py_routes) {
                // Use Python's str() function explicitly
                py::object str_func = py::module_::import("builtins").attr("str");
                std::string route_str = str_func(item).cast<std::string>();
                std::cout << "Vehicle " << v_idx++ << ": " << route_str << std::endl;
            }
            std::cout << "==========================================" << std::endl;
            
        } else {
            std::cerr << "[C++ Host] Optimization Failed or Status not found!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[C++ Host] Critical Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
