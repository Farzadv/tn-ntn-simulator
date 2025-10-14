"""
Coordinate-Based Geometry Utilities for TN-NTN Multi-Connectivity Simulator

This module provides comprehensive geometry calculations for all access types in the simulator.
All distances and angles are calculated from actual geographical coordinates (latitude, longitude, altitude)
ensuring realistic propagation modeling consistent with real-world deployments.

Key Features:
- Haversine distance calculations for Earth's curved surface
- 3D distance calculations accounting for altitude differences
- Elevation and azimuth angle computations for link budget analysis
- Support for terrestrial, UAV, HAPS, and satellite geometries
- Thread-safe coordinate transformations

The module uses standard geodetic calculations with WGS84 reference ellipsoid.
"""

import math
import yaml
from typing import Tuple


class GeometryUtils:
    """
    Comprehensive geometry utilities for TN-NTN multi-connectivity scenarios.
    
    This class provides methods to calculate distances, elevation angles, and azimuth angles
    between the UE and various access network nodes. All calculations use real geographical
    coordinates to ensure accuracy in path loss and link budget computations.
    
    The class supports:
    - Terrestrial base stations (typically at 30m altitude)
    - UAVs (typically at 300m altitude)
    - HAPS (typically at 20km altitude)
    - LEO satellites (typically at 550km altitude)
    
    Attributes:
        config: Loaded YAML configuration dictionary
        earth_radius_km: Earth radius in kilometers (from config)
        earth_radius_m: Earth radius in meters
        ue_position: UE location as [lat, lon, alt_meters]
        terrestrial_positions: List of terrestrial base station positions
        uav_positions: List of UAV positions
        haps_positions: List of HAPS positions
    """
    
    def __init__(self, config_path: str = "config/parameters.yaml"):
        """
        Initialize geometry utilities by loading configuration and Earth parameters.
        
        The initialization process:
        1. Loads the YAML configuration file containing all system parameters
        2. Extracts Earth radius for distance calculations
        3. Retrieves UE position from configuration
        4. Loads all access node positions (terrestrial, UAV, HAPS)
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If configuration file does not exist
            KeyError: If required parameters are missing from configuration
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Earth parameters from configuration
        # Used for all distance calculations and coordinate transformations
        self.earth_radius_km = self.config['constants']['earth_radius_km']
        self.earth_radius_m = self.earth_radius_km * 1000
        
        # UE position: [latitude_degrees, longitude_degrees, altitude_meters]
        self.ue_position = self.config['ue']['location']
        
        # Access node positions from configuration
        # Format: [[lat, lon, alt], ...] for each access type
        self.terrestrial_positions = self.config['terrestrial']['base_stations']
        self.uav_positions = self.config['uav']['positions']
        self.haps_positions = self.config['haps']['positions']
        
        print("Coordinate-based geometry initialized")
        print(f"UE position: {self.ue_position}")
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points on Earth using Haversine formula.
        
        The Haversine formula provides the shortest distance between two points on a sphere,
        accounting for Earth's curvature. This is critical for accurate distance calculations
        in wireless communications, especially for long-distance links.
        
        Mathematical formula:
            a = sin²(Δφ/2) + cos(φ1) × cos(φ2) × sin²(Δλ/2)
            c = 2 × atan2(√a, √(1-a))
            distance = R × c
        
        where:
            φ = latitude in radians
            λ = longitude in radians
            R = Earth's radius
            Δφ = lat2 - lat1
            Δλ = lon2 - lon1
        
        Args:
            lat1: Latitude of first point in decimal degrees
            lon1: Longitude of first point in decimal degrees
            lat2: Latitude of second point in decimal degrees
            lon2: Longitude of second point in decimal degrees
            
        Returns:
            Horizontal distance in meters along Earth's surface (great circle distance)
            
        Note:
            This method calculates only horizontal distance. For 3D distance including
            altitude differences, use calculate_3d_distance() method.
            
        Example:
            >>> geo = GeometryUtils()
            >>> # Distance from Paris to London
            >>> d = geo.haversine_distance(48.8566, 2.3522, 51.5074, -0.1278)
            >>> print(f"Distance: {d/1000:.1f} km")
        """
        # Convert decimal degrees to radians for trigonometric functions
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate coordinate differences
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad
        
        # Haversine formula implementation
        # a represents the square of half the chord length between the points
        a = (math.sin(delta_lat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        
        # c is the angular distance in radians
        c = 2 * math.asin(math.sqrt(a))
        
        # Multiply by Earth's radius to get actual distance
        horizontal_distance = self.earth_radius_m * c
        
        return horizontal_distance
    
    def calculate_3d_distance(self, pos1: Tuple[float, float, float],
                             pos2: Tuple[float, float, float]) -> float:
        """
        Calculate true 3D distance between two points considering both horizontal distance and altitude.
        
        This method combines the Haversine horizontal distance with the altitude difference
        using the Pythagorean theorem to compute the actual 3D Euclidean distance. This is
        the distance used in free-space path loss calculations.
        
        Mathematical approach:
            1. Calculate horizontal distance using Haversine formula
            2. Calculate altitude difference (vertical component)
            3. Apply Pythagorean theorem: d_3d = √(d_horizontal² + Δalt²)
        
        Args:
            pos1: First position as (latitude, longitude, altitude) 
                  Units: (degrees, degrees, meters)
            pos2: Second position as (latitude, longitude, altitude)
                  Units: (degrees, degrees, meters)
            
        Returns:
            3D Euclidean distance in meters
            
        Note:
            This is the distance that should be used for:
            - Free space path loss (FSPL) calculations
            - Link budget analysis
            - Propagation delay computations
            
        Example:
            >>> geo = GeometryUtils()
            >>> ue_pos = (48.8566, 2.3522, 0)      # Paris, ground level
            >>> haps_pos = (48.8566, 2.3522, 20000)  # HAPS directly above
            >>> d = geo.calculate_3d_distance(ue_pos, haps_pos)
            >>> print(f"3D distance: {d/1000:.1f} km")  # Should be ~20 km
        """
        lat1, lon1, alt1 = pos1
        lat2, lon2, alt2 = pos2
        
        # Calculate horizontal distance using Haversine formula
        # This accounts for Earth's curvature
        horizontal_distance = self.haversine_distance(lat1, lon1, lat2, lon2)
        
        # Calculate altitude difference (vertical component)
        altitude_difference = alt2 - alt1
        
        # Apply Pythagorean theorem for 3D distance
        # distance_3d = √(horizontal² + vertical²)
        distance_3d = math.sqrt(horizontal_distance**2 + altitude_difference**2)
        
        return distance_3d
    
    def calculate_elevation_angle(self, pos1: Tuple[float, float, float],
                                 pos2: Tuple[float, float, float]) -> float:
        """
        Calculate elevation angle from observer position (pos1) to target position (pos2).
        
        Elevation angle is critical for:
        - Antenna pointing and beam alignment
        - Path loss calculations (especially for NTN)
        - Line-of-sight (LoS) probability determination
        - Handover decisions in satellite systems
        
        The angle is calculated using the arctangent of the altitude difference
        over the horizontal distance.
        
        Formula:
            elevation = arctan(Δaltitude / horizontal_distance)
        
        Args:
            pos1: Observer position (latitude, longitude, altitude) in degrees, degrees, meters
            pos2: Target position (latitude, longitude, altitude) in degrees, degrees, meters
            
        Returns:
            Elevation angle in degrees
            - Positive values: target is above horizon
            - Negative values: target is below horizon
            - 0°: target is at horizon level
            - 90°: target is directly overhead (zenith)
            - -90°: target is directly below (nadir)
            
        Note:
            For satellite communications, minimum elevation angles are typically:
            - LEO: 20-25° (to ensure sufficient signal quality)
            - GEO: 5-10° (lower threshold due to higher power)
            
        Example:
            >>> geo = GeometryUtils()
            >>> ue_pos = (48.8566, 2.3522, 0)
            >>> sat_pos = (48.8566, 2.3522, 550000)  # LEO satellite above
            >>> elev = geo.calculate_elevation_angle(ue_pos, sat_pos)
            >>> print(f"Elevation: {elev:.1f}°")
        """
        lat1, lon1, alt1 = pos1
        lat2, lon2, alt2 = pos2
        
        # Calculate horizontal distance (ground distance)
        horizontal_distance = self.haversine_distance(lat1, lon1, lat2, lon2)
        
        # Calculate altitude difference (height difference)
        altitude_difference = alt2 - alt1
        
        # Handle special case: points at same horizontal location
        if horizontal_distance < 1e-6:  # Less than 1mm distance
            if altitude_difference > 0:
                return 90.0  # Target directly above
            elif altitude_difference < 0:
                return -90.0  # Target directly below
            else:
                return 0.0   # Same position
        
        # Calculate elevation angle using arctangent
        # atan2 handles the quadrant correctly and avoids division by zero
        elevation_radians = math.atan2(altitude_difference, horizontal_distance)
        elevation_degrees = math.degrees(elevation_radians)
        
        return elevation_degrees
    
    def calculate_azimuth_angle(self, pos1: Tuple[float, float, float],
                               pos2: Tuple[float, float, float]) -> float:
        """
        Calculate azimuth angle (bearing) from observer position to target position.
        
        Azimuth angle represents the horizontal direction from the observer to the target,
        measured clockwise from true North. This is essential for:
        - Directional antenna pointing
        - Beamforming and beam steering
        - Spatial diversity analysis
        - Interference coordination
        
        Convention:
            - 0° or 360°: North
            - 90°: East
            - 180°: South
            - 270°: West
        
        The calculation uses spherical trigonometry to account for Earth's curvature,
        which is important for accurate bearing calculations over long distances.
        
        Args:
            pos1: Observer position (latitude, longitude, altitude)
            pos2: Target position (latitude, longitude, altitude)
            
        Returns:
            Azimuth angle in degrees, range [0, 360)
            Measured clockwise from true North
            
        Note:
            Altitude is not used in azimuth calculation as azimuth is purely
            a horizontal angle. However, the tuple format is maintained for
            consistency with other geometry methods.
            
        Example:
            >>> geo = GeometryUtils()
            >>> paris = (48.8566, 2.3522, 0)
            >>> london = (51.5074, -0.1278, 0)
            >>> azimuth = geo.calculate_azimuth_angle(paris, london)
            >>> print(f"Bearing from Paris to London: {azimuth:.1f}°")
        """
        lat1, lon1, alt1 = pos1  # alt1 not used for azimuth
        lat2, lon2, alt2 = pos2  # alt2 not used for azimuth
        
        # Convert degrees to radians for trigonometric calculations
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Calculate longitude difference
        delta_lon = lon2_rad - lon1_rad
        
        # Azimuth calculation using spherical trigonometry
        # This accounts for Earth's curvature for accurate bearing
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        # atan2 returns angle in range [-π, π]
        azimuth_radians = math.atan2(y, x)
        azimuth_degrees = math.degrees(azimuth_radians)
        
        # Normalize to range [0, 360) degrees
        # Ensure azimuth is always positive (clockwise from North)
        if azimuth_degrees < 0:
            azimuth_degrees += 360.0
        
        return azimuth_degrees
    
    def get_terrestrial_geometry(self, ue_position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Calculate complete geometry parameters for terrestrial base station link.
        
        Terrestrial base stations are typically deployed at low altitudes (30-50m)
        for urban/suburban coverage. This method calculates all geometric parameters
        needed for terrestrial link budget analysis.
        
        Args:
            ue_position: UE location as (latitude, longitude, altitude_meters)
            
        Returns:
            Tuple of (distance_3d_m, elevation_deg, azimuth_deg)
            - distance_3d_m: 3D Euclidean distance in meters
            - elevation_deg: Elevation angle in degrees
            - azimuth_deg: Azimuth angle in degrees [0, 360)
            
        Note:
            Uses the first base station from configuration. For multi-BS scenarios,
            iterate through self.terrestrial_positions.
        """
        bs_position = self.terrestrial_positions[0]
        bs_pos_tuple = (bs_position[0], bs_position[1], bs_position[2])
        
        distance_3d = self.calculate_3d_distance(ue_position, bs_pos_tuple)
        elevation = self.calculate_elevation_angle(ue_position, bs_pos_tuple)
        azimuth = self.calculate_azimuth_angle(ue_position, bs_pos_tuple)
        
        return distance_3d, elevation, azimuth
    
    def get_uav_geometry(self, ue_position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Calculate complete geometry parameters for UAV link.
        
        UAVs typically operate at altitudes of 100-500m, providing temporary coverage
        or capacity enhancement. The geometry calculation considers the UAV's
        current position which may change over time.
        
        Args:
            ue_position: UE location as (latitude, longitude, altitude_meters)
            
        Returns:
            Tuple of (distance_3d_m, elevation_deg, azimuth_deg)
            
        Note:
            For mobile UAVs, the position should be updated regularly. This method
            uses the first UAV from configuration.
        """
        uav_position = self.uav_positions[0]
        uav_pos_tuple = (uav_position[0], uav_position[1], uav_position[2])
        
        distance_3d = self.calculate_3d_distance(ue_position, uav_pos_tuple)
        elevation = self.calculate_elevation_angle(ue_position, uav_pos_tuple)
        azimuth = self.calculate_azimuth_angle(ue_position, uav_pos_tuple)
        
        return distance_3d, elevation, azimuth
    
    def get_haps_geometry(self, ue_position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Calculate complete geometry parameters for HAPS link.
        
        HAPS operate at stratospheric altitudes (~20km), providing wide-area coverage
        with near-GEO characteristics but much lower latency. The high altitude results
        in typically high elevation angles and relatively stable geometry.
        
        Args:
            ue_position: UE location as (latitude, longitude, altitude_meters)
            
        Returns:
            Tuple of (distance_3d_m, elevation_deg, azimuth_deg)
            
        Note:
            HAPS typically maintain station-keeping within small regions, so geometry
            is relatively stable compared to LEO satellites.
        """
        haps_position = self.haps_positions[0]
        haps_pos_tuple = (haps_position[0], haps_position[1], haps_position[2])
        
        distance_3d = self.calculate_3d_distance(ue_position, haps_pos_tuple)
        elevation = self.calculate_elevation_angle(ue_position, haps_pos_tuple)
        azimuth = self.calculate_azimuth_angle(ue_position, haps_pos_tuple)
        
        return distance_3d, elevation, azimuth
    
    def get_satellite_geometry_from_state(self, ue_position: Tuple[float, float, float],
                                        satellite_state) -> Tuple[float, float, float]:
        """
        Calculate complete geometry parameters for satellite link from constellation state.
        
        This method interfaces with the constellation simulator to get real-time satellite
        positions and calculate the instantaneous geometry. LEO satellites move rapidly
        (~7.5 km/s orbital velocity), so geometry changes significantly over time.
        
        Args:
            ue_position: UE location as (latitude, longitude, altitude_meters)
            satellite_state: SatelliteState object from constellation simulator containing:
                - latitude: Satellite latitude in degrees
                - longitude: Satellite longitude in degrees
                - altitude: Satellite altitude in kilometers
                
        Returns:
            Tuple of (distance_3d_m, elevation_deg, azimuth_deg)
            
        Note:
            Satellite geometry changes rapidly. For LEO at 550km altitude:
            - Orbital period: ~95 minutes
            - Angular velocity: ~1°/second in UE's sky
            - Typical pass duration: 5-10 minutes above minimum elevation
        """
        sat_lat = satellite_state.latitude
        sat_lon = satellite_state.longitude
        sat_alt = satellite_state.altitude * 1000  # Convert km to meters
        
        sat_position = (sat_lat, sat_lon, sat_alt)
        
        distance_3d = self.calculate_3d_distance(ue_position, sat_position)
        elevation = self.calculate_elevation_angle(ue_position, sat_position)
        azimuth = self.calculate_azimuth_angle(ue_position, sat_position)
        
        return distance_3d, elevation, azimuth
    
    def get_all_access_geometries(self, ue_position: Tuple[float, float, float],
                                satellite_state=None) -> dict:
        """
        Calculate geometry parameters for all available access types simultaneously.
        
        This is a convenience method that computes complete geometry information for
        all access networks in a single call. This is useful for:
        - Comparative link budget analysis
        - Access network selection algorithms
        - Multi-connectivity scenarios
        - Energy efficiency comparisons
        
        Args:
            ue_position: UE location as (latitude, longitude, altitude_meters)
            satellite_state: Optional SatelliteState from constellation simulator
                           If None, satellite geometry will not be included
            
        Returns:
            Dictionary with geometry for each access type:
            {
                'terrestrial': {
                    'distance_3d_m': float,
                    'elevation_deg': float,
                    'azimuth_deg': float
                },
                'uav': {...},
                'haps': {...},
                'satellite': {...}  # Only if satellite_state provided
            }
            
        Example:
            >>> geo = GeometryUtils()
            >>> ue_pos = (48.8566, 2.3522, 0)
            >>> geometries = geo.get_all_access_geometries(ue_pos)
            >>> print(f"Terrestrial distance: {geometries['terrestrial']['distance_3d_m']:.0f} m")
            >>> print(f"HAPS elevation: {geometries['haps']['elevation_deg']:.1f}°")
        """
        geometries = {}
        
        # Calculate terrestrial base station geometry
        tn_distance, tn_elevation, tn_azimuth = self.get_terrestrial_geometry(ue_position)
        geometries['terrestrial'] = {
            'distance_3d_m': tn_distance,
            'elevation_deg': tn_elevation,
            'azimuth_deg': tn_azimuth
        }
        
        # Calculate UAV geometry
        uav_distance, uav_elevation, uav_azimuth = self.get_uav_geometry(ue_position)
        geometries['uav'] = {
            'distance_3d_m': uav_distance,
            'elevation_deg': uav_elevation,
            'azimuth_deg': uav_azimuth
        }
        
        # Calculate HAPS geometry
        haps_distance, haps_elevation, haps_azimuth = self.get_haps_geometry(ue_position)
        geometries['haps'] = {
            'distance_3d_m': haps_distance,
            'elevation_deg': haps_elevation,
            'azimuth_deg': haps_azimuth
        }
        
        # Calculate satellite geometry if satellite state is provided
        if satellite_state is not None:
            sat_distance, sat_elevation, sat_azimuth = self.get_satellite_geometry_from_state(
                ue_position, satellite_state
            )
            
            geometries['satellite'] = {
                'satellite_id': satellite_state.sat_id,
                'distance_3d_m': sat_distance,
                'elevation_deg': sat_elevation,
                'azimuth_deg': sat_azimuth
            }
        
        return geometries
