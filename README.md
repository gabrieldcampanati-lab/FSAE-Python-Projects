Phase 1: Foundations (Weeks 1-2)
Week 1: Python Refresher + Vehicle Dynamics Basics
Learning Goals:

Refresh Python fundamentals

Understand brake system physics

Set up your development environment

Resources:

Python Refresher:

Python.org Beginner's Guide - Official documentation, always reliable
​

Corey Schafer's YouTube tutorials - Excellent for visual learners
​

Real Python tutorials - Comprehensive and beginner-friendly
​

Vehicle Dynamics Foundation:

Mike Law's "Introduction to FSAE Vehicle Dynamics" - Watch this first! Covers tire behavior, bicycle model, weight transfer, and brake balance (47 minutes, incredibly valuable)
​

MIT FSAE Vehicle Dynamics thesis - Read sections on suspension geometry and weight transfer
​

Practical Tasks:

python
# Day 1-2: Environment Setup
# Install Anaconda (includes NumPy, Pandas, Matplotlib, SciPy)
# IDE: VSCode or Jupyter Notebook (recommend Jupyter for this project)

# Day 3-4: Basic calculations script
import numpy as np
import matplotlib.pyplot as plt

# Vehicle parameters (estimate for Formula Student car)
mass = 250  # kg (with driver)
wheelbase = 1.6  # meters
cg_height = 0.3  # meters
weight_dist_front = 0.48  # 48% front weight distribution

# Calculate static weight on each axle
weight_front = mass * 9.81 * weight_dist_front
weight_rear = mass * 9.81 * (1 - weight_dist_front)

print(f"Static front weight: {weight_front:.1f} N")
print(f"Static rear weight: {weight_rear:.1f} N")

# Day 5-7: Weight transfer during braking
def calculate_weight_transfer(mass, deceleration_g, cg_height, wheelbase):
    """
    Calculate longitudinal weight transfer during braking
    """
    weight_transfer = (mass * 9.81 * deceleration_g * cg_height) / wheelbase
    return weight_transfer

# Test with 1.5g braking (typical Formula Student max)
decels = np.linspace(0, 1.5, 20)
transfers = [calculate_weight_transfer(mass, d, cg_height, wheelbase) for d in decels]

plt.plot(decels, transfers)
plt.xlabel('Deceleration (g)')
plt.ylabel('Weight Transfer (N)')
plt.title('Longitudinal Weight Transfer During Braking')
plt.grid(True)
plt.show()
Physics Concepts to Master:

Kinetic energy to heat conversion: When braking from velocity v to 0, energy dissipated = 0.5 × m × v²

Weight transfer equation: ΔW_front = (m × a × h_cg) / L
​
​

Brake force distribution: Ideal distribution varies with deceleration due to weight transfer
​

Week 2: NumPy & Pandas Mastery
Learning Resources:

NumPy & Pandas Fast Tutorial - 30-minute video covering essentials
​

NumPy and Pandas for Data Analysis - Written tutorial with examples
​

Practical Tasks:

python
# Day 8-10: NumPy practice with vehicle dynamics
import numpy as np

# Create velocity profile array
time = np.linspace(0, 100, 1000)  # 100 seconds, 1000 points
velocity = 20 * np.sin(time/10) + 25  # Oscillating speed 5-45 m/s

# Calculate kinetic energy at each timestep
mass = 250
kinetic_energy = 0.5 * mass * velocity**2

# Find braking events (negative acceleration)
acceleration = np.gradient(velocity, time)
braking_mask = acceleration < -0.5  # Threshold for braking detection

print(f"Number of braking events detected: {np.sum(np.diff(braking_mask.astype(int)) > 0)}")

# Day 11-14: Pandas for data management
import pandas as pd

# Create DataFrame for lap data structure
lap_data = pd.DataFrame({
    'distance': np.linspace(0, 1000, 100),  # Track distance in meters
    'velocity': np.random.uniform(10, 40, 100),  # Placeholder velocities
    'curvature': np.random.uniform(0, 0.1, 100),  # Track curvature
    'braking': np.random.choice([0, 1], 100)  # Braking flag
})

# Calculate brake energy for each segment
lap_data['kinetic_energy'] = 0.5 * mass * lap_data['velocity']**2
lap_data['energy_delta'] = lap_data['kinetic_energy'].diff()
lap_data['brake_energy'] = -lap_data['energy_delta'] * lap_data['braking']

# Export for later use
lap_data.to_csv('lap_simulation_template.csv', index=False)
Deliverable: By end of Week 2, you should have:

✅ Development environment fully configured

✅ Basic vehicle dynamics calculations working

✅ Comfort with NumPy arrays and Pandas DataFrames

✅ Simple plots showing weight transfer and energy calculations

Phase 2: Core Simulator Development (Weeks 3-6)
Week 3-4: Track Model & Velocity Profile
Objective: Create a simplified model of Autódromo de Piracicaba and generate realistic velocity profiles.

Track Data Acquisition:

Since detailed Piracicaba data may be limited, you have two approaches:

Option A: Simplified Generic Track

python
# Create simplified track model (good for initial development)
import numpy as np
import pandas as pd

def create_simple_track(total_length=1000, num_corners=8):
    """
    Generate simplified track with straights and corners
    """
    segments = []
    distance = 0
    
    segment_length = total_length / (num_corners * 2)  # Alternating straight/corner
    
    for i in range(num_corners * 2):
        if i % 2 == 0:  # Straight
            segments.append({
                'start_distance': distance,
                'end_distance': distance + segment_length,
                'type': 'straight',
                'curvature': 0.0,
                'max_speed': 40  # m/s
            })
        else:  # Corner
            segments.append({
                'start_distance': distance,
                'end_distance': distance + segment_length,
                'type': 'corner',
                'curvature': np.random.uniform(0.05, 0.15),  # 1/m
                'max_speed': np.random.uniform(15, 25)  # m/s
            })
        
        distance += segment_length
    
    return pd.DataFrame(segments)

track = create_simple_track()
print(track.head())
Option B: Use TUM Lap Time Simulation Structure
​

The TUM repository provides production-grade structure:

Vehicle parameter files (.ini format)

Track raceline format with curvature

Quasi-steady-state velocity calculation

python
# Simplified TUM-inspired approach
def calculate_corner_speed(curvature, max_lateral_accel=1.5):
    """
    Calculate maximum corner speed based on curvature
    max_lateral_accel in g (typical FSAE: 1.3-1.8g)
    """
    if curvature < 0.001:  # Straight
        return 45  # Max speed in m/s
    
    # v = sqrt(a_lat / curvature), where a_lat = g * lateral_g
    max_speed = np.sqrt((max_lateral_accel * 9.81) / curvature)
    return min(max_speed, 45)  # Cap at reasonable max

# Apply to track
track['max_corner_speed'] = track['curvature'].apply(calculate_corner_speed)
Velocity Profile Generation:

This is crucial—you need realistic speed traces.
​

python
def generate_velocity_profile(track_data, vehicle_params):
    """
    Generate realistic velocity profile using forward-backward integration
    Based on OptimumLap methodology
    """
    distances = track_data['distance'].values
    max_speeds = track_data['max_corner_speed'].values
    
    n_points = len(distances)
    velocity = np.zeros(n_points)
    
    # Vehicle performance parameters
    max_accel = vehicle_params['max_accel']  # m/s²
    max_decel = vehicle_params['max_decel']  # m/s²
    
    # Forward pass: acceleration limited
    velocity[0] = max_speeds[0]  # Start at corner exit speed
    
    for i in range(1, n_points):
        ds = distances[i] - distances[i-1]
        
        # Accelerate as much as possible
        v_accel = np.sqrt(velocity[i-1]**2 + 2 * max_accel * ds)
        
        # But limited by corner entry speed
        velocity[i] = min(v_accel, max_speeds[i])
    
    # Backward pass: braking limited
    for i in range(n_points-2, -1, -1):
        ds = distances[i+1] - distances[i]
        
        # Calculate max speed we can brake from
        v_brake = np.sqrt(velocity[i+1]**2 + 2 * max_decel * ds)
        
        # Take minimum of acceleration and braking limit
        velocity[i] = min(velocity[i], v_brake)
    
    return velocity

vehicle_params = {
    'max_accel': 7.0,  # m/s² (typical FSAE with aero)
    'max_decel': 15.0  # m/s² (1.5g braking)
}

track['velocity'] = generate_velocity_profile(track, vehicle_params)
Week 5-6: Brake Energy & Thermal Calculations
Core Physics Implementation:

python
def calculate_brake_loads(velocity_profile, distance, vehicle_params):
    """
    Calculate brake energy dissipation at each point
    """
    # Calculate acceleration
    dt = 0.01  # Assume 100 Hz sampling
    accel = np.gradient(velocity_profile) / dt
    
    # Identify braking events
    braking = accel < -0.5  # Threshold for braking detection
    
    # Calculate brake force (F = m*a)
    brake_force = np.abs(accel) * vehicle_params['mass'] * braking
    
    # Split between front/rear based on weight transfer
    brake_force_front, brake_force_rear = calculate_brake_distribution(
        brake_force, accel, vehicle_params
    )
    
    # Calculate power dissipated (P = F*v)
    power_front = brake_force_front * velocity_profile
    power_rear = brake_force_rear * velocity_profile
    
    # Calculate energy per braking event
    energy_front = power_front * dt  # Joules
    energy_rear = power_rear * dt
    
    return {
        'brake_force_front': brake_force_front,
        'brake_force_rear': brake_force_rear,
        'power_front': power_front,
        'power_rear': power_rear,
        'energy_front': energy_front,
        'energy_rear': energy_rear,
        'braking_flag': braking
    }

def calculate_brake_distribution(total_force, decel, vehicle_params):
    """
    Calculate front/rear brake force distribution with weight transfer
    Based on: [artifact:web:60] MakerLuis brake distribution study
    """
    mass = vehicle_params['mass']
    wheelbase = vehicle_params['wheelbase']
    cg_height = vehicle_params['cg_height']
    static_front_pct = vehicle_params['weight_dist_front']
    
    # Weight transfer
    weight_transfer = (mass * np.abs(decel) * cg_height) / wheelbase
    
    # Dynamic weight on front axle
    weight_front_dyn = mass * 9.81 * static_front_pct + weight_transfer
    weight_total = mass * 9.81
    
    # Ideal distribution proportional to dynamic weight
    front_force_fraction = weight_front_dyn / weight_total
    
    brake_force_front = total_force * front_force_fraction
    brake_force_rear = total_force * (1 - front_force_fraction)
    
    return brake_force_front, brake_force_rear
Temperature Estimation (Lumped Capacitance Method):
​

python
def estimate_disc_temperature(energy_array, time_array, disc_params, ambient_temp=25):
    """
    Estimate brake disc temperature using lumped capacitance
    Based on: [artifact:web:64] Lumped-body thermal analysis
    """
    # Disc properties
    mass_disc = disc_params['mass']  # kg
    cp = disc_params['specific_heat']  # J/(kg·K), ~500 for cast iron
    area = disc_params['surface_area']  # m²
    
    # Convection parameters
    h_conv = disc_params['convection_coeff']  # W/(m²·K), estimate 50-150
    emissivity = disc_params['emissivity']  # ~0.8 for oxidized metal
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    
    temp = np.zeros(len(time_array))
    temp[0] = ambient_temp
    
    for i in range(1, len(time_array)):
        dt = time_array[i] - time_array[i-1]
        
        # Heat input from braking
        Q_in = energy_array[i]
        
        # Heat loss: convection + radiation
        Q_conv = h_conv * area * (temp[i-1] - ambient_temp) * dt
        Q_rad = emissivity * sigma * area * (temp[i-1]**4 - (ambient_temp+273)**4) * dt
        
        # Temperature change
        dT = (Q_in - Q_conv - Q_rad) / (mass_disc * cp)
        
        temp[i] = temp[i-1] + dT
    
    return temp

# Disc parameters (example for Formula Student)
disc_params = {
    'mass': 0.8,  # kg
    'specific_heat': 500,  # J/(kg·K)
    'surface_area': 0.025,  # m² (both sides of 200mm disc)
    'convection_coeff': 100,  # W/(m²·K) - varies with speed
    'emissivity': 0.8
}

# Calculate temperature profile
temp_front = estimate_disc_temperature(
    brake_loads['energy_front'],
    time_array,
    disc_params
)
Deliverable: By end of Week 6:

✅ Complete velocity profile generator

✅ Brake force calculator with front/rear distribution

✅ Energy dissipation computation

✅ Temperature estimation function

✅ Basic plotting of results

Phase 3: Validation & Integration (Weeks 7-8)
Week 7: Data Export & Visualization
Create outputs compatible with SolidWorks/ANSYS:

python
def export_thermal_loads(brake_data, filename='brake_thermal_loads.csv'):
    """
    Export brake loads in format suitable for FEA import
    """
    output_df = pd.DataFrame({
        'time_s': brake_data['time'],
        'distance_m': brake_data['distance'],
        'velocity_ms': brake_data['velocity'],
        'brake_force_front_N': brake_data['brake_force_front'],
        'brake_force_rear_N': brake_data['brake_force_rear'],
        'power_front_W': brake_data['power_front'],
        'power_rear_W': brake_data['power_rear'],
        'temp_front_C': brake_data['temp_front'],
        'temp_rear_C': brake_data['temp_rear'],
        'heat_flux_front_Wm2': brake_data['power_front'] / disc_params['surface_area'],
        'reynolds_number': calculate_reynolds_number(brake_data)
    })
    
    output_df.to_csv(filename, index=False)
    print(f"Exported {len(output_df)} timesteps to {filename}")
    
    return output_df

def calculate_reynolds_number(brake_data):
    """
    Calculate Reynolds number for convective cooling analysis
    Re = (rho * v * L) / mu
    """
    rho_air = 1.225  # kg/m³
    mu_air = 1.81e-5  # Pa·s
    char_length = 0.2  # m (disc diameter)
    
    velocity = brake_data['velocity']
    Re = (rho_air * velocity * char_length) / mu_air
    
    return Re
Comprehensive Visualization Dashboard:

python
import matplotlib.pyplot as plt

def plot_simulation_results(sim_data):
    """
    Create comprehensive visualization of simulation results
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # 1. Velocity profile
    axes[0, 0].plot(sim_data['distance'], sim_data['velocity'])
    axes[0, 0].set_xlabel('Distance (m)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].set_title('Velocity Profile Around Lap')
    axes[0, 0].grid(True)
    
    # 2. Brake force distribution
    axes[0, 1].plot(sim_data['distance'], sim_data['brake_force_front'], 
                     label='Front', linewidth=2)
    axes[0, 1].plot(sim_data['distance'], sim_data['brake_force_rear'], 
                     label='Rear', linewidth=2)
    axes[0, 1].set_xlabel('Distance (m)')
    axes[0, 1].set_ylabel('Brake Force (N)')
    axes[0, 1].set_title('Brake Force Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. Power dissipation
    axes[1, 0].plot(sim_data['time'], sim_data['power_front']/1000, 
                     label='Front', linewidth=2)
    axes[1, 0].plot(sim_data['time'], sim_data['power_rear']/1000, 
                     label='Rear', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Power (kW)')
    axes[1, 0].set_title('Instantaneous Power Dissipation')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Temperature evolution
    axes[1, 1].plot(sim_data['time'], sim_data['temp_front'], 
                     label='Front', linewidth=2)
    axes[1, 1].plot(sim_data['time'], sim_data['temp_rear'], 
                     label='Rear', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_title('Disc Temperature Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 5. Energy per braking event
    braking_events = identify_braking_events(sim_data)
    axes[2, 0].bar(range(len(braking_events)), 
                    [e['energy_front']/1000 for e in braking_events])
    axes[2, 0].set_xlabel('Braking Event')
    axes[2, 0].set_ylabel('Energy (kJ)')
    axes[2, 0].set_title('Energy per Braking Event (Front)')
    axes[2, 0].grid(True, axis='y')
    
    # 6. Reynolds number
    axes[2, 1].plot(sim_data['velocity'], sim_data['reynolds_number'], 'o')
    axes[2, 1].set_xlabel('Velocity (m/s)')
    axes[2, 1].set_ylabel('Reynolds Number')
    axes[2, 1].set_title('Reynolds Number vs Velocity')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('brake_simulation_results.png', dpi=300)
    plt.show()
Week 8: SolidWorks/ANSYS Integration
When your thermal simulation is ready:

Import simulation data into SolidWorks Flow Simulation or ANSYS Transient Thermal

Apply heat flux as boundary condition (calculated from your simulator)

Compare predicted vs simulated temperatures

Iterate on convection coefficient estimation to match results

Validation workflow:
​

python
def compare_simulation_to_fea(simulator_temps, fea_temps, time_array):
    """
    Compare Python simulator predictions to FEA results
    """
    # Calculate error metrics
    mae = np.mean(np.abs(simulator_temps - fea_temps))
    rmse = np.sqrt(np.mean((simulator_temps - fea_temps)**2))
    max_error = np.max(np.abs(simulator_temps - fea_temps))
    
    print(f"Mean Absolute Error: {mae:.2f}°C")
    print(f"Root Mean Square Error: {rmse:.2f}°C")
    print(f"Maximum Error: {max_error:.2f}°C")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, simulator_temps, label='Python Simulator', linewidth=2)
    plt.plot(time_array, fea_temps, label='SolidWorks FEA', 
             linewidth=2, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('Simulator vs FEA Validation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {'mae': mae, 'rmse': rmse, 'max_error': max_error}
Phase 4: Optimization & Team Value (Weeks 9-10)
Week 9: Parameter Studies
Make your tool useful for design decisions:

python
def parameter_sensitivity_study(base_params, param_name, param_range):
    """
    Study effect of parameter changes on brake performance
    Useful for design optimization
    """
    results = []
    
    for param_value in param_range:
        # Update parameter
        test_params = base_params.copy()
        test_params[param_name] = param_value
        
        # Run simulation
        sim_result = run_brake_simulation(test_params)
        
        results.append({
            param_name: param_value,
            'max_temp': np.max(sim_result['temp_front']),
            'avg_temp': np.mean(sim_result['temp_front'][sim_result['braking_flag']]),
            'total_energy': np.sum(sim_result['energy_front'])
        })
    
    return pd.DataFrame(results)

# Example: Study disc mass effect
disc_masses = np.linspace(0.6, 1.2, 10)
mass_study = parameter_sensitivity_study(base_params, 'disc_mass', disc_masses)

plt.plot(mass_study['disc_mass'], mass_study['max_temp'])
plt.xlabel('Disc Mass (kg)')
plt.ylabel('Maximum Temperature (°C)')
plt.title('Effect of Disc Mass on Peak Temperature')
plt.grid(True)
plt.show()
Week 10: Documentation & Team Presentation
Create professional documentation:
​
​

README.md explaining how to use your simulator

User guide with example cases

Technical report explaining physics and validation

Team presentation showing how it helps brake design
