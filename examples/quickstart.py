from vehicle_state_estimation.experiments import run

if __name__ == "__main__":
    output = run(seed=7, steps=100)
    print(f"quickstart results: {output}")
