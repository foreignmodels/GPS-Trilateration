import numpy as np
import matplotlib.pyplot as plt

# Receiver coordinates
receiver_x = np.random.randint(-100, 101)
receiver_y = np.random.randint(-100, 101)

# Satellite coordinates
satellite1_x = np.random.randint(-100, 101)
satellite1_y = np.random.randint(-100, 101)

satellite2_x = np.random.randint(-100, 101)
satellite2_y = np.random.randint(-100, 101)

satellite3_x = np.random.randint(-100, 101)
satellite3_y = np.random.randint(-100, 101)

# 2D Distances
distance1 = np.sqrt((receiver_x - satellite1_x)**2 + (receiver_y - satellite1_y)**2)
distance2 = np.sqrt((receiver_x - satellite2_x)**2 + (receiver_y - satellite2_y)**2)
distance3 = np.sqrt((receiver_x - satellite3_x)**2 + (receiver_y - satellite3_y)**2)

# Simulated signal travel times
speed_of_light = 299_792.458 # km/s
time1 = distance1 / speed_of_light
time2 = distance2 / speed_of_light
time3 = distance3 / speed_of_light

# Measured distances from satellites to receiver
measured1 = speed_of_light * time1
measured2 = speed_of_light * time2
measured3 = speed_of_light * time3

# Satellite positions
satellites = np.array([
    [satellite1_x, satellite1_y],
    [satellite2_x, satellite2_y],
    [satellite3_x, satellite3_y]
], dtype=float)

# Measured distances
measured = np.array([measured1, measured2, measured3])

# 2D Trilateration equations
A = 2 * (satellites[1:] - satellites[0])
b = measured[0]**2 - measured[1:]**2
b += np.sum(satellites[1:]**2, axis=1) - np.sum(satellites[0]**2)
calculated = np.linalg.lstsq(A, b, rcond=None)[0]

# Matplotlib graph
plt.figure(figsize=(8, 8))

colors = ["blue", "red", "green"]

for i in range(3):
    plt.gca().add_patch(plt.Circle(satellites[i], measured[i],
                        fill=False, color=colors[i], linewidth=2))
    plt.scatter(satellites[i, 0], satellites[i, 1],
                color=colors[i], marker="^", s=80,
                label=f"Satellite {i + 1}")

# Receiver location
plt.scatter(calculated[0], calculated[1],
            color="black", marker="o", s=100,
            label="Calculated Receiver")

# Graph
plt.xlim(-350, 350)
plt.ylim(-350, 350)
plt.gca().set_aspect("equal")
plt.xlabel("X Position (km)")
plt.ylabel("Y Position (km)")
plt.title("GPS Trilateration Simulation")
plt.grid(True, alpha=0.3)
plt.legend()

# Results
print("\nSatellite Distances:")
print(f"  Satellite 1: {measured1:.2f} km")
print(f"  Satellite 2: {measured2:.2f} km")
print(f"  Satellite 3: {measured3:.2f} km")

print("\nReceiver Coordinates:")
print(f"  Actual:     ({receiver_x}, {receiver_y})")
print(f"  Calculated: ({calculated[0]:.2f}, {calculated[1]:.2f})")

plt.show()