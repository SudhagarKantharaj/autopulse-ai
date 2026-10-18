
from pathlib import Path
import random
import pandas as pd

random.seed(42)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

rows = []

for vehicle_id in ["EV-1001", "EV-1002", "EV-1003"]:
    for reading in range(100):
        soc = random.randint(20, 95)
        battery_temp = round(random.uniform(22, 38), 1)
        motor_temp = round(random.uniform(30, 75), 1)
        voltage = round(random.uniform(350, 405), 1)
        dtc = "NONE"

        # Synthetic battery temperature fault
        if vehicle_id == "EV-1002" and reading >= 90:
            battery_temp = round(random.uniform(47, 53), 1)
            dtc = "BATT_TEMP_HIGH"

        # Synthetic motor temperature fault
        if vehicle_id == "EV-1003" and reading >= 90:
            motor_temp = round(random.uniform(92, 105), 1)
            dtc = "MOTOR_TEMP_HIGH"

        rows.append({
            "vehicle_id": vehicle_id,
            "reading_number": reading + 1,
            "battery_soc_pct": soc,
            "battery_temp_c": battery_temp,
            "motor_temp_c": motor_temp,
            "pack_voltage_v": voltage,
            "dtc": dtc,
        })

df = pd.DataFrame(rows)
df.to_csv(DATA_DIR / "vehicle_telemetry.csv", index=False)

print(f"Created {len(df)} telemetry records.")
print(df.head())
print("\nRecords per vehicle:")
print(df.groupby("vehicle_id").size())
