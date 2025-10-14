"""
LEO Constellation Simulator with Automatic Position Updates

Provides realistic LEO satellite constellation simulation with:
- Automatic satellite position updates in background thread
- Keplerian orbital mechanics
- Satellite visibility and handover management
- Thread-safe state management

Supports Starlink and Kuiper constellation configurations.
"""

import numpy as np
import datetime
import yaml
import threading
import time
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
from enum import Enum


class HandoverMetric(Enum):
    """Metrics for satellite handover decisions"""
    ELEVATION_ANGLE = "elevation_angle"
    SNR = "snr"
    RSRP = "rsrp"
    DISTANCE = "distance"


@dataclass
class SatelliteState:
    """Complete state information for a satellite at a given time"""
    sat_id: int
    plane_id: int
    sat_in_plane: int
    latitude: float
    longitude: float
    altitude: float  # km
    velocity: Tuple[float, float, float]  # m/s in ECEF frame
    elevation_angle: float  # degrees from UE perspective
    azimuth_angle: float   # degrees from UE perspective
    distance_3d: float     # km from UE
    snr_db: Optional[float] = None
    rsrp_dbm: Optional[float] = None


@dataclass
class HandoverState:
    """Tracks UE-satellite connection state for handover management"""
    current_satellite_id: Optional[int] = None
    connection_start_time: Optional[datetime.datetime] = None
    handover_count: int = 0
    last_handover_time: Optional[datetime.datetime] = None


class ConstellationSimulator:
    """
    LEO Constellation Simulator with automatic position updates.
    
    Simulates LEO satellite constellations (Starlink, Kuiper) with realistic
    orbital mechanics. Satellites are initialized in their orbital planes
    according to constellation parameters, and positions are updated automatically
    in a background thread to simulate orbital motion.
    
    Key features:
    - Automatic position updates (default: every 5 seconds)
    - Thread-safe satellite state management
    - Elevation-based satellite selection and handover
    - Support for multiple UE connections
    """
    
    def __init__(self, config_path: str = "config/parameters.yaml"):
        """
        Initialize constellation simulator and start automatic updates.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.constellation_type = self.config['satellite']['constellation_type']
        self.satellite_config = self.config['satellite'][self.constellation_type]
        self.general_sat_config = self.config['satellite']
        
        self.earth_radius = self.config['constants']['earth_radius_km']
        
        # Time management with thread safety
        self.current_time = datetime.datetime.utcnow()
        self.time_lock = threading.RLock()
        self.update_interval = self.config['simulation']['time_step']
        
        # Auto-update control
        self.auto_update_enabled = False
        self.auto_update_thread = None
        self._stop_auto_update = threading.Event()
        
        # Handover configuration
        self.handover_config = self.config.get('handover', {})
        self.handover_threshold = self.handover_config.get('elevation_threshold_deg', 20.0)
        self.handover_metrics = [HandoverMetric(m) for m in self.handover_config.get('metrics', ['elevation_angle'])]
        self.min_visibility = self.handover_config.get('min_visibility_threshold_deg', 5.0)
        
        # UE connection tracking (thread-safe)
        self.ue_connections = {}
        self.connections_lock = threading.Lock()
        
        # Initialize constellation
        self._initialize_constellation()
        
        print(f"Constellation simulator initialized:")
        print(f"  Type: {self.constellation_type}")
        print(f"  Total satellites: {self.total_satellites}")
        print(f"  Update interval: {self.update_interval}s")
        
        # Auto-start position updates
        self.start_auto_update()
    
    def _initialize_constellation(self):
        """
        Initialize satellite constellation with Keplerian orbital elements.
        
        Creates satellites distributed across orbital planes with proper phasing.
        Each satellite is initialized with:
        - Orbital plane (RAAN - Right Ascension of Ascending Node)
        - Position in plane (Mean Anomaly)
        - Orbital parameters (inclination, altitude)
        """
        self.num_planes = self.satellite_config['num_planes']
        self.sats_per_plane = self.satellite_config['sats_per_plane']
        self.inclination = math.radians(self.satellite_config['inclination_deg'])
        self.altitude = self.general_sat_config['altitude_km']
        self.total_satellites = self.num_planes * self.sats_per_plane
        
        # Orbital parameters
        self.semi_major_axis = self.earth_radius + self.altitude
        self.orbital_period = self._calculate_orbital_period()
        self.mean_motion = 2 * math.pi / self.orbital_period
        
        # Initialize satellites
        self.satellites = []
        sat_id = 0
        
        for plane_idx in range(self.num_planes):
            # RAAN: evenly distributed around Earth
            raan = (plane_idx * 360.0 / self.num_planes) % 360.0
            
            for sat_idx in range(self.sats_per_plane):
                # Mean anomaly: evenly distributed in orbital plane
                mean_anomaly = (sat_idx * 360.0 / self.sats_per_plane) % 360.0
                
                satellite = {
                    'sat_id': sat_id,
                    'plane_id': plane_idx,
                    'sat_in_plane': sat_idx,
                    'raan_deg': raan,
                    'inclination_deg': math.degrees(self.inclination),
                    'mean_anomaly_deg': mean_anomaly,
                    'epoch_time': self.current_time,
                    'eccentricity': 0.0,  # Circular orbits
                    'argument_of_perigee_deg': 0.0
                }
                
                self.satellites.append(satellite)
                sat_id += 1
    
    def _calculate_orbital_period(self) -> float:
        """
        Calculate orbital period using Kepler's third law.
        
        Formula: T = 2π√(a³/μ)
        where a is semi-major axis and μ is Earth's gravitational parameter
        
        Returns:
            Orbital period in seconds
        """
        mu_earth = 398600.4418  # km³/s² (Earth's gravitational parameter)
        period = 2 * math.pi * math.sqrt(self.semi_major_axis**3 / mu_earth)
        return period
    
    def start_auto_update(self) -> None:
        """
        Start automatic position updates in background thread.
        
        The update loop increments simulation time by update_interval
        every update_interval seconds, simulating satellite orbital motion.
        """
        if self.auto_update_enabled:
            return
        
        self.auto_update_enabled = True
        self._stop_auto_update.clear()
        
        def update_loop():
            try:
                while not self._stop_auto_update.is_set():
                    # Thread-safe time update
                    with self.time_lock:
                        self.current_time += datetime.timedelta(seconds=self.update_interval)
                    
                    # Sleep for update interval
                    if not self._stop_auto_update.wait(self.update_interval):
                        continue
                    else:
                        break
                        
            except Exception as e:
                print(f"Auto-update engine error: {e}")
        
        self.auto_update_thread = threading.Thread(target=update_loop, daemon=True, name="SatelliteUpdater")
        self.auto_update_thread.start()
    
    def stop_auto_update(self) -> None:
        """Stop automatic position updates"""
        if not self.auto_update_enabled:
            return
        
        self._stop_auto_update.set()
        self.auto_update_enabled = False
        
        if self.auto_update_thread and self.auto_update_thread.is_alive():
            self.auto_update_thread.join(timeout=2.0)
    
    def get_current_simulation_time(self) -> datetime.datetime:
        """Get current simulation time (thread-safe)"""
        with self.time_lock:
            return self.current_time
    
    def _satellite_position_ecef(self, satellite: Dict, time_offset: float) -> Tuple[float, float, float]:
        """
        Calculate satellite position in ECEF (Earth-Centered Earth-Fixed) coordinates.
        
        Uses simplified Keplerian propagation for circular orbits.
        
        Args:
            satellite: Satellite orbital elements dictionary
            time_offset: Time elapsed since epoch (seconds)
            
        Returns:
            (x, y, z) position in ECEF frame (kilometers)
        """
        mean_anomaly_rad = math.radians(satellite['mean_anomaly_deg'])
        mean_anomaly_current = mean_anomaly_rad + self.mean_motion * time_offset
        true_anomaly = mean_anomaly_current  # For circular orbits, true anomaly ≈ mean anomaly
        
        raan_rad = math.radians(satellite['raan_deg'])
        inc_rad = math.radians(satellite['inclination_deg'])
        
        # Position in orbital plane
        r = self.semi_major_axis
        x_orb = r * math.cos(true_anomaly)
        y_orb = r * math.sin(true_anomaly)
        z_orb = 0.0
        
        # Rotation from orbital plane to ECEF
        cos_raan = math.cos(raan_rad)
        sin_raan = math.sin(raan_rad)
        cos_inc = math.cos(inc_rad)
        sin_inc = math.sin(inc_rad)
        
        x_ecef = cos_raan * x_orb - sin_raan * cos_inc * y_orb
        y_ecef = sin_raan * x_orb + cos_raan * cos_inc * y_orb
        z_ecef = sin_inc * y_orb
        
        return x_ecef, y_ecef, z_ecef
    
    def _ecef_to_lla(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert ECEF coordinates to latitude, longitude, altitude.
        
        Uses iterative method for geodetic coordinate conversion.
        
        Args:
            x, y, z: ECEF coordinates in kilometers
            
        Returns:
            (latitude_deg, longitude_deg, altitude_km)
        """
        x_m, y_m, z_m = x * 1000, y * 1000, z * 1000
        
        # WGS84 ellipsoid parameters
        a = 6378137.0  # Semi-major axis (meters)
        e2 = 6.69437999014e-3  # First eccentricity squared
        
        # Longitude is straightforward
        longitude = math.degrees(math.atan2(y_m, x_m))
        
        # Latitude requires iteration
        p = math.sqrt(x_m**2 + y_m**2)
        lat = math.atan2(z_m, p * (1 - e2))
        
        # Iterate to refine latitude and altitude
        for _ in range(5):
            N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
            altitude_m = p / math.cos(lat) - N
            lat = math.atan2(z_m, p * (1 - e2 * N / (N + altitude_m)))
        
        latitude = math.degrees(lat)
        altitude_km = altitude_m / 1000.0
        
        return latitude, longitude, altitude_km
    
    def _calculate_look_angles(self, ue_lat: float, ue_lon: float, ue_alt: float,
                             sat_lat: float, sat_lon: float, sat_alt: float) -> Tuple[float, float, float]:
        """
        Calculate elevation, azimuth, and distance from UE to satellite.
        
        Transforms both positions to ECEF, calculates vector difference,
        then converts to local topocentric coordinates (East-North-Up).
        
        Args:
            ue_lat, ue_lon, ue_alt: UE position (degrees, degrees, km)
            sat_lat, sat_lon, sat_alt: Satellite position (degrees, degrees, km)
            
        Returns:
            (elevation_deg, azimuth_deg, distance_3d_km)
        """
        # Convert to radians
        ue_lat_rad = math.radians(ue_lat)
        ue_lon_rad = math.radians(ue_lon)
        sat_lat_rad = math.radians(sat_lat)
        sat_lon_rad = math.radians(sat_lon)
        
        # Convert to ECEF
        earth_radius_ue = self.earth_radius + ue_alt
        earth_radius_sat = self.earth_radius + sat_alt
        
        ue_x = earth_radius_ue * math.cos(ue_lat_rad) * math.cos(ue_lon_rad)
        ue_y = earth_radius_ue * math.cos(ue_lat_rad) * math.sin(ue_lon_rad)
        ue_z = earth_radius_ue * math.sin(ue_lat_rad)
        
        sat_x = earth_radius_sat * math.cos(sat_lat_rad) * math.cos(sat_lon_rad)
        sat_y = earth_radius_sat * math.cos(sat_lat_rad) * math.sin(sat_lon_rad)
        sat_z = earth_radius_sat * math.sin(sat_lat_rad)
        
        # Vector from UE to satellite
        dx = sat_x - ue_x
        dy = sat_y - ue_y
        dz = sat_z - ue_z
        
        distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Local topocentric frame (East-North-Up)
        east_x = -math.sin(ue_lon_rad)
        east_y = math.cos(ue_lon_rad)
        east_z = 0.0
        
        north_x = -math.sin(ue_lat_rad) * math.cos(ue_lon_rad)
        north_y = -math.sin(ue_lat_rad) * math.sin(ue_lon_rad)
        north_z = math.cos(ue_lat_rad)
        
        up_x = math.cos(ue_lat_rad) * math.cos(ue_lon_rad)
        up_y = math.cos(ue_lat_rad) * math.sin(ue_lon_rad)
        up_z = math.sin(ue_lat_rad)
        
        # Project onto local frame
        east_component = dx * east_x + dy * east_y + dz * east_z
        north_component = dx * north_x + dy * north_y + dz * north_z
        up_component = dx * up_x + dy * up_y + dz * up_z
        
        # Calculate elevation and azimuth
        horizontal_distance = math.sqrt(east_component**2 + north_component**2)
        elevation_rad = math.atan2(up_component, horizontal_distance)
        elevation_deg = math.degrees(elevation_rad)
        
        azimuth_rad = math.atan2(east_component, north_component)
        azimuth_deg = math.degrees(azimuth_rad)
        if azimuth_deg < 0:
            azimuth_deg += 360.0
        
        return elevation_deg, azimuth_deg, distance_3d
    
    def get_satellite_states(self, ue_position: Tuple[float, float, float]) -> List[SatelliteState]:
        """
        Get current states of all satellites relative to UE position.
        
        Args:
            ue_position: UE location as (lat, lon, alt_meters)
            
        Returns:
            List of SatelliteState objects for all satellites
        """
        ue_lat, ue_lon, ue_alt_m = ue_position
        ue_alt_km = ue_alt_m / 1000.0
        
        satellite_states = []
        
        # Get current time (thread-safe)
        with self.time_lock:
            current_time = self.current_time
        
        current_timestamp = current_time.timestamp()
        
        for satellite in self.satellites:
            epoch_timestamp = satellite['epoch_time'].timestamp()
            time_offset = current_timestamp - epoch_timestamp
            
            # Calculate satellite position
            sat_x, sat_y, sat_z = self._satellite_position_ecef(satellite, time_offset)
            sat_lat, sat_lon, sat_alt = self._ecef_to_lla(sat_x, sat_y, sat_z)
            
            # Calculate look angles
            elevation, azimuth, distance_3d = self._calculate_look_angles(
                ue_lat, ue_lon, ue_alt_km, sat_lat, sat_lon, sat_alt
            )
            
            # Approximate orbital velocity
            orbital_velocity = 2 * math.pi * self.semi_major_axis * 1000 / self.orbital_period
            
            state = SatelliteState(
                sat_id=satellite['sat_id'],
                plane_id=satellite['plane_id'],
                sat_in_plane=satellite['sat_in_plane'],
                latitude=sat_lat,
                longitude=sat_lon,
                altitude=sat_alt,
                velocity=(0, orbital_velocity, 0),  # Simplified
                elevation_angle=elevation,
                azimuth_angle=azimuth,
                distance_3d=distance_3d
            )
            
            satellite_states.append(state)
        
        return satellite_states
    
    def get_serving_satellite(self, ue_position: Tuple[float, float, float],
                            ue_id: str = "default") -> Optional[SatelliteState]:
        """
        Get serving satellite using handover logic with elevation-based selection.
        
        Handover logic:
        - Stay connected to current satellite if elevation > threshold
        - Switch to satellite with highest elevation when current drops below threshold
        - Hysteresis prevents ping-pong handovers
        
        Args:
            ue_position: UE location (lat, lon, alt_meters)
            ue_id: Unique UE identifier for connection tracking
            
        Returns:
            SatelliteState of serving satellite, or None if no satellite available
        """
        # Thread-safe access to UE connections
        with self.connections_lock:
            if ue_id not in self.ue_connections:
                self.ue_connections[ue_id] = HandoverState()
            handover_state = self.ue_connections[ue_id]
        
        satellite_states = self.get_satellite_states(ue_position)
        
        # Filter visible satellites (above minimum elevation)
        visible_satellites = [s for s in satellite_states if s.elevation_angle >= self.min_visibility]
        
        if not visible_satellites:
            with self.connections_lock:
                if handover_state.current_satellite_id is not None:
                    handover_state.current_satellite_id = None
            return None
        
        # Check if current satellite still meets threshold
        current_satellite = None
        if handover_state.current_satellite_id is not None:
            current_satellite = next(
                (s for s in visible_satellites if s.sat_id == handover_state.current_satellite_id),
                None
            )
            
            # Stay connected if current satellite meets threshold
            if current_satellite is not None and current_satellite.elevation_angle >= self.handover_threshold:
                return current_satellite
        
        # Find best satellite (highest elevation)
        best_satellite = max(visible_satellites, key=lambda s: s.elevation_angle)
        
        # Check if best satellite meets minimum threshold
        if best_satellite.elevation_angle < self.handover_threshold:
            with self.connections_lock:
                handover_state.current_satellite_id = None
            return None
        
        # Perform handover if needed (thread-safe)
        with self.connections_lock:
            with self.time_lock:
                current_time = self.current_time
            
            old_satellite_id = handover_state.current_satellite_id
            handover_state.current_satellite_id = best_satellite.sat_id
            handover_state.connection_start_time = current_time
            
            if old_satellite_id != best_satellite.sat_id:
                handover_state.handover_count += 1
                handover_state.last_handover_time = current_time
        
        return best_satellite
    
    def get_handover_statistics(self, ue_id: str = "default") -> Dict:
        """
        Get handover statistics for a specific UE.
        
        Args:
            ue_id: UE identifier
            
        Returns:
            Dictionary with handover statistics
        """
        with self.connections_lock:
            if ue_id not in self.ue_connections:
                return {"error": "UE not found", "handover_count": 0}
            
            state = self.ue_connections[ue_id]
            
            with self.time_lock:
                current_time = self.current_time
            
            connection_duration = None
            if state.connection_start_time:
                connection_duration = (current_time - state.connection_start_time).total_seconds()
            
            return {
                "current_satellite_id": state.current_satellite_id,
                "handover_count": state.handover_count,
                "last_handover_time": state.last_handover_time.isoformat() if state.last_handover_time else None,
                "current_connection_duration_seconds": connection_duration,
                "auto_update_enabled": self.auto_update_enabled
            }
    
    def get_constellation_info(self) -> Dict:
        """Get constellation configuration information"""
        with self.time_lock:
            current_time = self.current_time
        
        return {
            'constellation_type': self.constellation_type,
            'total_satellites': self.total_satellites,
            'num_planes': self.num_planes,
            'sats_per_plane': self.sats_per_plane,
            'altitude_km': self.altitude,
            'inclination_deg': math.degrees(self.inclination),
            'orbital_period_minutes': self.orbital_period / 60.0,
            'handover_threshold': self.handover_threshold,
            'handover_metrics': [m.value for m in self.handover_metrics],
            'auto_update_enabled': self.auto_update_enabled,
            'current_simulation_time': current_time.isoformat(),
            'update_interval_seconds': self.update_interval
        }
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_auto_update()
        except:
            pass
