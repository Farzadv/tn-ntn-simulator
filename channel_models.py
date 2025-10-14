"""
Channel Models for TN-NTN Multi-Connectivity Simulator

Implements path loss calculations for terrestrial and non-terrestrial networks.
Includes 3GPP-compliant models for terrestrial networks and NTN-specific propagation models
for UAV, HAPS, and satellite links.
"""

import numpy as np
import math
import yaml
from typing import Tuple
from enum import Enum


class AccessType(Enum):
    """Enumeration of different access types"""
    TERRESTRIAL = "terrestrial"
    UAV = "uav"
    HAPS = "haps"
    SATELLITE = "satellite"


class ChannelModels:
    """
    Channel models for calculating path loss across different access types.
    
    Implements:
    - Free space path loss (FSPL) for baseline calculations
    - 3GPP TR 38.901 models for terrestrial networks
    - Probabilistic LoS models for UAV air-to-ground channels
    - Stratospheric propagation models for HAPS
    - LEO satellite channel models with atmospheric effects
    """
    
    def __init__(self, config_path: str = "config/parameters.yaml"):
        """
        Initialize channel models with configuration parameters.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.speed_of_light = self.config['constants']['speed_of_light']
        
        print("Channel models initialized")
    
    def free_space_path_loss(self, frequency_hz: float, distance_3d_m: float) -> float:
        """
        Calculate free space path loss (FSPL).
        
        FSPL represents the baseline path loss in ideal conditions without obstacles.
        Formula: FSPL(dB) = 20*log10(4*π*f*d/c)
        
        Args:
            frequency_hz: Carrier frequency in Hz
            distance_3d_m: 3D distance in meters
            
        Returns:
            Path loss in dB
        """
        if distance_3d_m <= 0:
            return 0.0
        
        fspl_db = 20 * math.log10(4 * math.pi * frequency_hz * distance_3d_m / self.speed_of_light)
        
        return fspl_db
    
    def terrestrial_path_loss(self, distance_3d_m: float, frequency_hz: float,
                            environment: str = "UMa", los_condition: bool = True,
                            add_shadowing: bool = True) -> float:
        """
        Calculate terrestrial path loss using 3GPP TR 38.901 formulas.
        
        Supports three environment types:
        - UMa (Urban Macro): Dense urban environments with large cells
        - UMi (Urban Micro): Urban environments with small cells
        - RMa (Rural Macro): Rural/suburban environments
        
        Args:
            distance_3d_m: 3D distance in meters
            frequency_hz: Carrier frequency in Hz
            environment: Environment type (UMa, UMi, RMa)
            los_condition: True for LoS, False for NLoS
            add_shadowing: Whether to add random log-normal shadowing
            
        Returns:
            Path loss in dB
        """
        frequency_ghz = frequency_hz / 1e9
        
        # 3GPP minimum distance constraint
        if distance_3d_m < 10.0:
            distance_3d_m = 10.0
        
        # 3GPP TR 38.901 path loss formulas
        if environment == "UMa":
            if los_condition:
                # Urban Macro LoS
                path_loss = (32.4 +
                           21 * math.log10(distance_3d_m) +
                           20 * math.log10(frequency_ghz))
            else:
                # Urban Macro NLoS
                path_loss = (13.54 +
                           39.08 * math.log10(distance_3d_m) +
                           20 * math.log10(frequency_ghz))
                
        elif environment == "UMi":
            if los_condition:
                # Urban Micro LoS
                path_loss = (32.4 +
                           22 * math.log10(distance_3d_m) +
                           20 * math.log10(frequency_ghz))
            else:
                # Urban Micro NLoS
                path_loss = (22.7 +
                           36.7 * math.log10(distance_3d_m) +
                           26 * math.log10(frequency_ghz))
                
        else:  # RMa (Rural Macro)
            if los_condition:
                # Rural Macro LoS
                path_loss = (32.4 +
                           20 * math.log10(distance_3d_m) +
                           20 * math.log10(frequency_ghz))
            else:
                # Rural Macro NLoS
                path_loss = (32.4 +
                           30 * math.log10(distance_3d_m) +
                           20 * math.log10(frequency_ghz))
        
        # Add random log-normal shadowing
        if add_shadowing:
            shadowing_std = self.config['terrestrial']['shadowing_std_db']
            random_shadowing = np.random.normal(0, shadowing_std)
            path_loss += random_shadowing
        
        return path_loss
    
    def uav_path_loss(self, distance_3d_m: float, elevation_angle_deg: float,
                     frequency_hz: float, environment: str = "suburban",
                     add_shadowing: bool = True) -> float:
        """
        Calculate UAV air-to-ground path loss using probabilistic LoS model.
        
        The model accounts for elevation-dependent LoS probability:
        - Higher elevation angles → Higher LoS probability → Lower path loss
        - Lower elevation angles → More obstacles → Higher NLoS probability
        
        Args:
            distance_3d_m: 3D distance to UAV in meters
            elevation_angle_deg: Elevation angle from UE to UAV in degrees
            frequency_hz: Carrier frequency in Hz
            environment: Environment type (suburban, urban)
            add_shadowing: Whether to add random shadowing
            
        Returns:
            Path loss in dB
        """
        uav_config = self.config['uav']
        
        # Get environment-specific LoS probability parameters
        if environment in uav_config['environment_params']:
            env_params = uav_config['environment_params'][environment]
            a_env = env_params['a_env']
            b_env = env_params['b_env']
        else:
            # Default to suburban parameters
            a_env = uav_config['environment_params']['suburban']['a_env']
            b_env = uav_config['environment_params']['suburban']['b_env']
        
        # Calculate LoS probability as function of elevation angle
        # P_LoS = 1 / (1 + a*exp(-b*(θ - a)))
        p_los = 1 / (1 + a_env * math.exp(-b_env * (elevation_angle_deg - a_env)))
        p_los = max(0.0, min(1.0, p_los))  # Clamp to [0,1]
        
        # Free space path loss (baseline)
        fspl = self.free_space_path_loss(frequency_hz, distance_3d_m)
        
        # LoS/NLoS excess losses from configuration
        eta_los = uav_config['los_excess_loss_db']
        eta_nlos = uav_config['nlos_excess_loss_db']
        
        # Combined path loss with probabilistic LoS/NLoS
        path_loss = fspl + p_los * eta_los + (1 - p_los) * eta_nlos
        
        # Add random shadowing
        if add_shadowing:
            shadowing_std = uav_config['shadowing_std_db']
            random_shadowing = np.random.normal(0, shadowing_std)
            path_loss += random_shadowing
        
        return path_loss
    
    def haps_path_loss(self, distance_3d_m: float, elevation_angle_deg: float,
                      frequency_hz: float, add_shadowing: bool = True) -> float:
        """
        Calculate HAPS path loss using elevation-dependent LoS model.
        
        HAPS at ~20km altitude typically have high LoS probability due to
        stratospheric altitude clearing most atmospheric obstacles.
        
        Model: PL = p_LoS * PL^LoS + (1 - p_LoS) * PL^NLoS
        where PL^α = FSPL + CL^α (clutter loss)
        
        Args:
            distance_3d_m: 3D slant distance to HAPS in meters
            elevation_angle_deg: Elevation angle from UE to HAPS in degrees
            frequency_hz: Carrier frequency in Hz
            add_shadowing: Whether to add random log-normal shadowing
            
        Returns:
            Path loss in dB
        """
        haps_config = self.config['haps']
        
        # Get environment parameters for LoS probability model
        env_params = haps_config.get('environment_params', {
            'a_env': 4.88,
            'b_env': 0.43
        })
        a_env = env_params['a_env']
        b_env = env_params['b_env']
        
        # Calculate LoS probability using logistic model
        p_los = 1.0 / (1.0 + a_env * math.exp(-b_env * (elevation_angle_deg - a_env)))
        p_los = max(0.0, min(1.0, p_los))
        
        # Free space path loss (dominant component for HAPS)
        fspl = self.free_space_path_loss(frequency_hz, distance_3d_m)
        
        # Clutter losses from configuration
        cl_los = haps_config.get('clutter_loss_los_db', 0.0)
        cl_nlos = haps_config.get('clutter_loss_nlos_db', 20.0)
        
        # Path loss components
        pl_los = fspl + cl_los
        pl_nlos = fspl + cl_nlos
        
        # Combined path loss
        path_loss = p_los * pl_los + (1 - p_los) * pl_nlos
        
        # Add log-normal shadowing
        if add_shadowing:
            shadowing_std = haps_config.get('shadowing_std_db', 4.0)
            random_shadowing = np.random.normal(0, shadowing_std)
            path_loss += random_shadowing
        
        return path_loss
    
    def satellite_path_loss(self, distance_3d_m: float, elevation_angle_deg: float,
                          frequency_hz: float, add_shadowing: bool = True) -> float:
        """
        Calculate LEO satellite path loss.
        
        For LEO satellites at 550+ km altitude:
        - Free space path loss dominates (160+ dB)
        - Atmospheric absorption is relatively small
        - Ionospheric effects are minimal at L/S-band frequencies
        
        Args:
            distance_3d_m: 3D distance to satellite in meters
            elevation_angle_deg: Elevation angle (used for future enhancements)
            frequency_hz: Carrier frequency in Hz
            add_shadowing: Whether to add random shadowing
            
        Returns:
            Path loss in dB
        """
        satellite_config = self.config['satellite']
        
        # Free space path loss (primary component for satellites)
        fspl = self.free_space_path_loss(frequency_hz, distance_3d_m)
        
        # Atmospheric loss (relatively constant for LEO at sub-6 GHz)
        atmospheric_loss = satellite_config['atmospheric_loss_db']
        
        # Total deterministic path loss
        deterministic_pl = fspl + atmospheric_loss
        
        # Add random shadowing
        if add_shadowing:
            shadowing_std = satellite_config['shadowing_std_db']
            random_shadowing = np.random.normal(0, shadowing_std)
            deterministic_pl += random_shadowing
        
        return deterministic_pl
    
    def calculate_path_loss(self, access_type: AccessType, distance_3d_m: float,
                          elevation_angle_deg: float, frequency_hz: float,
                          add_shadowing: bool = True, **kwargs) -> float:
        """
        Unified interface to calculate path loss for any access type.
        
        This method provides a single entry point for path loss calculations
        across all access types, automatically dispatching to the appropriate
        model based on the access_type parameter.
        
        Args:
            access_type: Type of access (TERRESTRIAL, UAV, HAPS, SATELLITE)
            distance_3d_m: 3D distance in meters
            elevation_angle_deg: Elevation angle in degrees
            frequency_hz: Carrier frequency in Hz
            add_shadowing: Whether to add random shadowing (default: True)
            **kwargs: Additional parameters (environment, los_condition)
            
        Returns:
            Path loss in dB
        """
        if access_type == AccessType.TERRESTRIAL:
            environment = kwargs.get('environment', 'UMa')
            los_condition = kwargs.get('los_condition', True)
            return self.terrestrial_path_loss(distance_3d_m, frequency_hz,
                                            environment, los_condition, add_shadowing)
        
        elif access_type == AccessType.UAV:
            environment = kwargs.get('environment', 'suburban')
            return self.uav_path_loss(distance_3d_m, elevation_angle_deg,
                                    frequency_hz, environment, add_shadowing)
        
        elif access_type == AccessType.HAPS:
            return self.haps_path_loss(distance_3d_m, elevation_angle_deg,
                                     frequency_hz, add_shadowing)
        
        elif access_type == AccessType.SATELLITE:
            return self.satellite_path_loss(distance_3d_m, elevation_angle_deg,
                                          frequency_hz, add_shadowing)
        
        else:
            raise ValueError(f"Unknown access type: {access_type}")
    
    def calculate_noise_power(self, bandwidth_hz: float, noise_figure_db: float = None) -> float:
        """
        Calculate thermal noise power at receiver.
        
        Noise power is fundamental to SNR calculations and link budget analysis.
        Formula: N = N0 + 10*log10(BW) + NF
        where N0 is thermal noise density (-174 dBm/Hz at room temperature)
        
        Args:
            bandwidth_hz: Channel bandwidth in Hz
            noise_figure_db: Receiver noise figure in dB (from config if None)
            
        Returns:
            Noise power in dBm
        """
        if noise_figure_db is None:
            noise_figure_db = self.config['channel']['noise_figure_db']
        
        thermal_noise_dbm_hz = self.config['channel']['thermal_noise_density_dbm_hz']
        
        noise_power_dbm = (thermal_noise_dbm_hz +
                          10 * math.log10(bandwidth_hz) +
                          noise_figure_db)
        
        return noise_power_dbm
    
    def get_frequency_and_bandwidth(self, access_type: AccessType) -> Tuple[float, float]:
        """
        Get operating frequency and bandwidth from configuration.
        
        Each access type operates at different frequency bands:
        - Terrestrial: 3.5 GHz (5G NR n78 band)
        - UAV: 2.1 GHz (3GPP Band 1)
        - HAPS: 2.1 GHz (IMT bands)
        - Satellite: 1.995 GHz (S-band MSS)
        
        Args:
            access_type: Type of access
            
        Returns:
            Tuple of (frequency_hz, bandwidth_hz)
        """
        access_configs = {
            AccessType.TERRESTRIAL: 'terrestrial',
            AccessType.UAV: 'uav',
            AccessType.HAPS: 'haps',
            AccessType.SATELLITE: 'satellite'
        }
        
        if access_type not in access_configs:
            raise ValueError(f"Unknown access type: {access_type}")
        
        config_key = access_configs[access_type]
        freq_ghz = self.config[config_key]['frequency_ghz']
        bw_mhz = self.config[config_key]['bandwidth_mhz']
        
        return freq_ghz * 1e9, bw_mhz * 1e6
