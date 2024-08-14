import numpy as np
import matplotlib.pyplot as plt
import argparse
import json


def read_config():
    with open("config/config.json", "r") as f:
        config = json.load(f)
    return config


def read_residuals(solver):
    if solver == "steady_state":
        filename = "relative_residuals.dat"
    elif solver == "runge_kutta":
        filename = "residuals.dat"
    else:
        raise Exception(f"Unknown solver {solver}")
    with open(f"log/{filename}", "r") as f:
        headers = [x.strip() for x in f.readline().split(" ")]

    residuals_data = np.loadtxt(f"log/{filename}", delimiter=" ", skiprows=1)

    residuals = {}
    if solver == "runge_kutta":
        residuals["time"] = residuals_data[:, headers.index("time")]
    residuals["step"] = residuals_data[:, headers.index("step")]
    residuals["global"] = residuals_data[:, headers.index("global")]
    residuals["mass"] = residuals_data[:, headers.index("mass")]
    residuals["momentum_x"] = residuals_data[:, headers.index("momentum_x")]
    residuals["momentum_y"] = residuals_data[:, headers.index("momentum_y")]
    residuals["momentum_z"] = residuals_data[:, headers.index("momentum_z")]
    residuals["energy"] = residuals_data[:, headers.index("energy")]
    return residuals


def plot_residuals(residuals, x_axis):
    if x_axis == "time":
        x_values = residuals["time"]
    elif x_axis == "step":
        x_values = residuals["step"]
    else:
        raise Exception(f"Unknown x_axis {x_axis}")

    fig, ax = plt.subplots()
    ax.plot(x_values, residuals["global"], label="global")
    ax.plot(x_values, residuals["mass"], label="mass")
    ax.plot(x_values, residuals["momentum_x"], label="momentum_x")
    ax.plot(x_values, residuals["momentum_y"], label="momentum_y")
    ax.plot(x_values, residuals["momentum_z"], label="momentum_z")
    ax.plot(x_values, residuals["energy"], label="energy")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel(x_axis)
    ax.grid()
    plt.show()


def main():
    config = read_config()
    solver = config["solver"]["name"]
    if solver == "runge_kutta":
        default_x_axis = "time"
    elif solver == "steady_state":
        default_x_axis = "step"
    else:
        raise Exception(f"Unknown solver {solver}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_axis", default=default_x_axis)

    args = parser.parse_args()

    residuals = read_residuals(solver)
    plot_residuals(residuals, args.x_axis)


if __name__ == "__main__":
    main()
