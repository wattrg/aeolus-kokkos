add_library(
    simulation 
    STATIC 
    simulation/simulation.cpp
)
target_link_libraries(
    simulation
    PUBLIC
    finite_volume
    gas
    grid
    nlohmann_json::nlohmann_json
)
target_include_directories(simulation PUBLIC .)
