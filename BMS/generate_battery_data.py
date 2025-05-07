import csv
import random
from datetime import datetime, timedelta

# File name for the generated data
output_file = "dummy_battery_data.csv"

# Number of rows to generate
num_rows = 999

# Generate random alphanumeric IDs
def generate_id(prefix, num_digits=4):
    return f"{prefix}{random.randint(10**(num_digits-1), 10**num_digits-1)}"

# Generate random timestamp
def generate_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

# Generate dummy data
def generate_dummy_data():
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 1, 1)
    data = []
    for _ in range(num_rows):
        battery_id = generate_id("BAT")
        cell_id = generate_id("CELL")
        voltage = round(random.uniform(3.0, 4.2), 2)  # Voltage between 3.0 and 4.2
        current = round(random.uniform(0.5, 2.0), 2)  # Current between 0.5 and 2.0
        resistance = round(voltage / current, 2) if current != 0 else None
        soc = random.randint(50, 100)  # State of Charge between 50% and 100%
        sod = 100 - soc  # State of Discharge
        soh = random.randint(90, 100)  # State of Health between 90% and 100%
        cell_temperature = round(random.uniform(32,42), 2)  # cell_temperature between 32 and 42
        ambient_temperature = round(random.uniform(32,46), 2)  # ambient_temperature between 32 and 46
        timestamp = generate_timestamp(start_date, end_date).isoformat()
        state_of_cell = random.choice(["live", "dead"])  # State of the cell
        communication = random.choice(["yes", "no"])  # Communication of the cell
        data.append([battery_id, cell_id, voltage, current, resistance, soc, sod, soh, cell_temperature, ambient_temperature, timestamp, state_of_cell, communication])
    return data

# Write to CSV
def write_to_csv(data):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow([
            "Battery ID", "Cell ID", "Voltage (V)", "Current (A)",
            "Resistance (Ohms)", "SOC (%)", "SOD (%)", "SOH (%)",
            "Cell Temperature (°C)", "Ambient Temperature (°C)",
            "Timestamp", "State of Cell", "Communication"
        ])
        # Write data
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    dummy_data = generate_dummy_data()
    write_to_csv(dummy_data)
    print(f"Dummy data generated in {output_file}")