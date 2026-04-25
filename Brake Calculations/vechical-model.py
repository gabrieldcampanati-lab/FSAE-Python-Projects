#Vehicle parameters
class VehicleParameters:
    """
    Stores all vehicle parameters for FSAE brake simulator.
    
    Based on actual Pegasus Formula Student car specifications.
    References: Brake Design and Safety, Chapter 7
    """
    def __init__(self):
        
        # === Vehicle Geometry ===
        self.mass = 240  # kg (with driver)
        self.wheelbase = 1.53  # meters
        self.cg_height = 0.3  # meters
        
        # === Weight Distribution ===
        # Static weight distribution: fraction of weight on rear axle
        self.static_load_rear_ratio = 1 - 0.49  # β = 0.51 (51% rear, 49% front)
        
        # === Tire Parameters ===
        self.wheel_diameter = 0.254 # meters (10 inches)
        "Add more tire parameters as needed"
        
        # === Brake System Parameters ===
        self.master_cylinder_diameter = 0.0159 #meters
        self.master_cylindre_radius = self.master_cylinder_diameter / 2 #meters
        self.force_on_pedal = 823  # N
        self.pedal_mechanical_advantage = 0.8 #From BDS Chapter 7
        self.perdal_ratio = 4.4 #From P02 car specs
        self.brake_factor = 1 #From BDS Chapter 7
        
        # === Constants ===
        self.gravity = 9.81  # m/s^2
        self.total_weight = self.mass * self.gravity  # N
        
    #Properties: computed values based on parameters
    
    @property
    def static_load_front_ratio(self):
        #Fraction of weight on front axle
        return 1 - self.static_load_rear_ratio 
    
    @property
    def master_cylinder_area(self):
        #Cross-sectional area of master cylinder
        import math
        return math.pi * (self.master_cylindre_radius ** 2) #m^2
    
    @property
    def static_load_rear(self):
        #Static weight on rear axle
        return self.total_weight * self.static_load_rear_ratio #N
    
    @property
    def static_load_front(self):
        #Static weight on front axle
        return self.total_weight * self.static_load_front_ratio #N
    
    def get_summary(self):
        #Prints a summary of vehicle parameters
        print("=" * 60)
        print("Vehicle Parameters Summary:")
        print("=" * 60)
        print(f"\nGeometry:")
        print(f"Mass (with driver): {self.mass} kg")
        print(f"Total Weight: {self.total_weight:.1f} N")
        print(f"Wheelbase: {self.wheelbase} m")
        print(f"Center of Gravity Height: {self.cg_height} m")
        print(f"\nWeight Distribution:")
        print(f"Static Load Front: {self.static_load_front:.1f} N")
        print(f"Static Load Rear: {self.static_load_rear:.1f} N")
        print(f"\nBrake System:")
        print(f"Master Cylinder Area: {self.master_cylinder_area:.6f} m^2")
        print(f"Force on Pedal: {self.force_on_pedal} N")
        print(f"Pedal Ratio: {self.perdal_ratio}")
        print(f"Master Cylinder Diameter: {self.master_cylinder_diameter} m")
        print(f"\nTire Parameters:")
        print(f"Wheel Diameter: {self.wheel_diameter} m")
        print("=" * 60)
        
# Example usage
# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Create instance
    vehicle = VehicleParameters()
    
    # Access parameters
    print(f"Vehicle mass: {vehicle.mass} kg")
    print(f"Static rear load: {vehicle.static_load_rear:.1f} N")
    print(f"MC area: {vehicle.master_cylinder_area*1e4:.2f} cm²")
    
    # Print full summary
    vehicle.get_summary()