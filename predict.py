
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/ev_range_model.pkl")

print("\n====================================")
print("       EV BATTERY RANGE PREDICTOR")
print("====================================")

# Get input from user
make = input("Manufacturer (Tata): ")
model_name = input("EV Model (Nexon EV): ")
variant = input("Variant (Long Range): ")

battery_capacity = float(input("Battery Capacity (kWh): "))
battery_capacity_new = float(input("New Battery Capacity (kWh): "))
motor_power = int(input("Motor Power (kW): "))
vehicle_weight = int(input("Vehicle Weight (kg): "))

drag_coefficient = float(input("Drag Coefficient: "))
frontal_area = float(input("Frontal Area (m2): "))
rolling_resistance = float(input("Rolling Resistance: "))

motor_efficiency = int(input("Motor Efficiency (%): "))
arai_range = int(input("ARAI Certified Range (km): "))

avg_speed = float(input("Average Speed (km/h): "))
temperature = float(input("Temperature (C): "))

ac_usage = int(input("AC Usage (0 or 1): "))
terrain = int(input("Terrain Type (0-2): "))
traffic = int(input("Traffic Condition (0-2): "))
driving_style = int(input("Driving Style (0-2): "))

passengers = int(input("Number of Passengers: "))
battery_age = float(input("Battery Age (years): "))
tyre_pressure = int(input("Tyre Pressure Deviation (psi): "))

city_road_type = input("City Road Type: ")
price = float(input("Vehicle Price (lakh INR): "))

# Prepare input as DataFrame
input_data = pd.DataFrame([{
    "make": make,
    "model": model_name,
    "variant": variant,
    "battery_capacity_kWh": battery_capacity,
    "battery_capacity_kWh_new": battery_capacity_new,
    "motor_power_kW": motor_power,
    "vehicle_weight_kg": vehicle_weight,
    "drag_coefficient_Cd": drag_coefficient,
    "frontal_area_m2": frontal_area,
    "rolling_resistance_Cr": rolling_resistance,
    "motor_efficiency_pct": motor_efficiency,
    "ARAI_certified_range_km": arai_range,
    "avg_driving_speed_kmh": avg_speed,
    "ambient_temperature_C": temperature,
    "ac_usage": ac_usage,
    "terrain_type": terrain,
    "traffic_condition": traffic,
    "driving_style": driving_style,
    "passengers": passengers,
    "battery_age_years": battery_age,
    "tyre_pressure_deviation_psi": tyre_pressure,
    "city_road_type": city_road_type,
    "price_lakh_INR": price
}])

# Predict range
predicted_range = model.predict(input_data)

# Display prediction
print("\n====================================")
print("          PREDICTION RESULT")
print("====================================")
print("EV Manufacturer:", make)
print("EV Model:", model_name)
print("Predicted Real-World Range:",
      round(predicted_range[0], 2), "km")
print("====================================")
