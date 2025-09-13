import subprocess

# Fixed list of seeds
seeds = [
    2765,
    29209,
    44960,
    45245,
    60061,
    63420,
    76764,
    77245,
    80468,
    82034,
]

for seed in seeds:
    cmd = [
        "python",
        "train.py",
        "-config",
        "agents/MountainCar-v0/ppo.yaml",
        f"--seed_id={seed}"
    ]
    print(f"Running with seed {seed}...")
    subprocess.run(cmd)
