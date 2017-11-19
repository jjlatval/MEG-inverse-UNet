# Run this file to simulate data: python simulate_data.py

from __future__ import unicode_literals
from simulation_model.simulation_model import SimulationModel
from config import SIMULATION_MODEL_KWARGS


def main():
    sim = SimulationModel(**SIMULATION_MODEL_KWARGS)
    sim.save_simulated_data()

if __name__ == "__main__":
    main()
