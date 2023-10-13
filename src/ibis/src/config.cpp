#include <fstream>
#include <spdlog/spdlog.h>
#include "config.h"

json read_directories() {
    std::string ibis = std::getenv("IBIS");
    std::ifstream f(ibis + "/resources/defaults/directories.json");
    json directories = json::parse(f);
    f.close();
    return directories;
}

json read_config(json& directories) {
    // read the config file
    std::string config_dir = directories.at("config_dir");
    std::string config_file = directories.at("config_file");
    std::ifstream f(config_dir + "/" + config_file);
    if (!f) {
        spdlog::error("Unable to open config file. Make sure simulation is prepped");
        throw std::runtime_error("Unable to open config file");
    }
    json config = json::parse(f);
    f.close();
    return config;
}
