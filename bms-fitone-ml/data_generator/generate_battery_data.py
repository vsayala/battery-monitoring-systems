import csv
from datetime import datetime, timedelta
import random

# File name for the generated data
output_file = "dummy_battery_data.csv"

# Number of battery IDs to generate
num_batteries = 4

# Number of cells per battery
cells_per_battery = 12

# Number of records per cell (trend data over time)
records_per_cell = 99999

# Generate random alphanumeric IDs
def generate_id(prefix, num_digits=4):
    return f"{prefix}{random.randint(10**(num_digits-1), 10**num_digits-1)}"

# Generate dummy data
def generate_dummy_data():
    start_date = datetime(2023, 1, 1, 0, 0, 0)  # Start date and time

    # Ensure records_per_cell is valid for generating cell lifespan
    if records_per_cell < 500:
        raise ValueError(f"records_per_cell must be at least 500. Current value: {records_per_cell}")

    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow([
            "Battery ID", "Cell ID", "Voltage (V)", "Current (A)",
            "Resistance (Ohms)", "SOC (%)", "SOD (%)", "SOH (%)",
            "Cell Temperature (°C)", "Ambient Temperature (°C)",
            "Timestamp", "State of Cell", "Communication"
        ])

        for _ in range(num_batteries):
            battery_id = generate_id("BAT")  # Generate unique Battery ID

            # Ambient temperature for the battery (changes only when the timestamp changes)
            ambient_temperature = round(random.uniform(32, 46), 2)

            for _ in range(cells_per_battery):
                cell_id = generate_id("CELL")
                # Randomized threshold for transitioning to "dead" state
                cell_lifespan = random.randint(500, records_per_cell)
                is_dead = False  # Track if the cell is currently in the "dead" state
                current_timestamp = start_date  # Start timestamp for this cell

                for record_index in range(records_per_cell):
                    # Handle communication failure with a small probability for "no"
                    communication = random.choices(["yes", "no"], weights=[0.95, 0.05])[0]
                    if communication == "no":
                        # Populate nulls for all fields except communication and timestamp
                        writer.writerow([
                            battery_id, cell_id, None, None, None, None, None, None,
                            None, None, current_timestamp.isoformat(), None, communication
                        ])
                    else:
                        # Generate normal data
                        voltage = round(random.uniform(3.0, 4.2), 2)
                        current = round(random.uniform(0.5, 2.0), 2)
                        resistance = round(voltage / current, 2) if current != 0 else None
                        soc = random.randint(50, 100)
                        sod = 100 - soc
                        soh = random.randint(90, 100)
                        cell_temperature = round(random.uniform(32, 42), 2)

                        # Determine if the cell transitions to the "dead" state
                        if not is_dead and record_index >= cell_lifespan:
                            is_dead = True  # Transition to "dead" state

                        # If the cell is in the "dead" state, generate values that keep it "dead"
                        if is_dead:
                            state_of_cell = "dead"
                            voltage = round(random.uniform(2.5, 3.0), 2)
                            current = round(random.uniform(0.0, 0.5), 2)
                            resistance = round(voltage / (current + 0.1), 2)
                            soh = random.randint(10, 30)
                            cell_temperature = round(random.uniform(45, 50), 2)
                        else:
                            state_of_cell = "live"

                        # Append data
                        writer.writerow([
                            battery_id, cell_id, voltage, current, resistance, soc, sod, soh,
                            cell_temperature, ambient_temperature, current_timestamp.isoformat(),
                            state_of_cell, communication
                        ])

                    # Increment timestamp by 1 minute
                    current_timestamp += timedelta(minutes=1)

if __name__ == "__main__":
    generate_dummy_data()
    print(f"Dummy data generated in {output_file}")