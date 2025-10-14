"""
UE Power Consumption Models with Transport Block Level Analysis

Implements energy consumption calculations for UE in TN-NTN multi-connectivity scenarios.
Includes TB-level power calculations based on 5G NR parameters with accurate MCS-based
transport block sizing.

Key Features:
- Required transmit power calculations for link budget
- Energy per bit and energy per TB metrics
- Support for both power-constrained and SNR-target analysis modes
- 5G NR MCS table and TBS calculations (3GPP TS 38.214)
- Realistic satellite reference energy via constellation sampling
"""

import numpy as np
import math
import yaml
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
from channel_models import AccessType, ChannelModels


@dataclass
class PowerResult:
    """Results of power calculation with TB-level approach"""
    tx_power_dbm: float
    tx_power_linear_mw: float
    feasible: bool
    snr_achieved_db: float
    analysis_mode: str
    required_power_dbm: Optional[float] = None
    power_deficit_db: Optional[float] = None
    energy_per_tb_uj: float = 0.0
    tbs_bits: int = 0
    mcs_index: int = 0
    
    @property
    def energy_per_bit_uj(self) -> float:
        """Derive energy per bit from TB-level calculation"""
        if self.tbs_bits > 0:
            return self.energy_per_tb_uj / self.tbs_bits
        return 0.0


@dataclass
class EnergyAnalysis:
    """Energy analysis results with TB-level approach"""
    ul_power: PowerResult
    dl_power: PowerResult
    energy_saving_ratio: float
    lifetime_extension_factor: float
    analysis_mode: str
    total_energy_per_tb_uj: float = 0.0
    avg_tbs_bits: float = 0.0
    tb_transmission_rate_hz: float = 0.0
    
    @property
    def total_energy_per_bit_uj(self) -> float:
        """Derive total energy per bit from TB-level calculation"""
        if self.avg_tbs_bits > 0:
            return self.total_energy_per_tb_uj / self.avg_tbs_bits
        return 0.0


@dataclass
class MCSEntry:
    """MCS table entry for TB-level calculations (3GPP TS 38.214)"""
    mcs_index: int
    qm: int  # Modulation order (2=QPSK, 4=16QAM, 6=64QAM)
    r: float  # Code rate x 1024
    spectral_efficiency: float


class UEPowerModels:
    """
    UE Power Models with Transport Block Level Analysis.
    
    Provides comprehensive power consumption calculations for UE devices in
    multi-connectivity scenarios. Supports both traditional energy-per-bit
    metrics and 5G NR transport block level analysis.
    
    Analysis Modes:
    - power_constrained: Respects UE maximum TX power (23 dBm for Class 3)
    - snr_target: Calculates power needed to achieve target SNR
    """
    
    def __init__(self, config_path: str = "config/parameters.yaml", constellation=None):
        """
        Initialize power models with configuration and optional constellation.
        
        Args:
            config_path: Path to YAML configuration file
            constellation: Optional ConstellationSimulator for dynamic satellite reference
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.channel_models = ChannelModels(config_path)
        self.constellation = constellation
        
        # UE parameters
        self.ue_config = self.config['ue']
        self.max_tx_power_dbm = self.ue_config['max_tx_power_dbm']
        self.ue_antenna_gain_dbi = self.ue_config['antenna_gain_dbi']
        self.circuit_power_mw = self.ue_config['circuit_power_mw']
        self.rx_power_mw = self.ue_config['rx_power_mw']
        
        self.channel_config = self.config['channel']
        self.noise_figure_db = self.channel_config['noise_figure_db']
        
        # Application semantics
        app_semantics = self.config.get('applications_semantics', {})
        self.rates_are_instantaneous = app_semantics.get('rates_are_instantaneous_during_on', True)
        
        # Initialize 5G NR TB-level parameters
        self._init_5g_nr_tb_parameters()
        
        print("UE power models initialized - TB-Level analysis enabled")
        print(f"  5G NR: μ={self.nr_numerology}, T_slot={self.slot_duration_ms}ms")
        print(f"  Default MCS: UL={self.default_mcs_ul}, DL={self.default_mcs_dl}")
    
    def _init_5g_nr_tb_parameters(self):
        """Initialize 5G NR TB-level parameters from configuration"""
        nr_config = self.config.get('5g_nr', {})
        
        # Numerology and slot duration
        self.nr_numerology = nr_config.get('numerology_index', 1)
        slot_durations = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.125, 4: 0.0625}
        self.slot_duration_ms = slot_durations.get(self.nr_numerology, 0.5)
        
        # Override with config if provided
        if 'slot_duration_ms' in nr_config:
            self.slot_duration_ms = nr_config['slot_duration_ms']
        
        # Resource allocation
        self.nrb_ul = nr_config.get('nrb_ul', 50)
        self.nrb_dl = nr_config.get('nrb_dl', 50)
        self.nsym_data_ul = nr_config.get('nsym_data_ul', 12)
        self.nsym_data_dl = nr_config.get('nsym_data_dl', 11)
        
        # Default MCS indices
        self.default_mcs_ul = nr_config.get('mcs_ul', 10)
        self.default_mcs_dl = nr_config.get('mcs_dl', 15)
        
        # Initialize MCS table
        self._init_mcs_table()
    
    def _init_mcs_table(self):
        """Initialize 5G NR MCS table (3GPP TS 38.214 Table 5.1.3.1-1)"""
        self.mcs_table = [
            MCSEntry(0, 2, 120, 2*120/1024),    # QPSK
            MCSEntry(1, 2, 157, 2*157/1024),
            MCSEntry(2, 2, 193, 2*193/1024),
            MCSEntry(3, 2, 251, 2*251/1024),
            MCSEntry(4, 2, 308, 2*308/1024),
            MCSEntry(5, 2, 379, 2*379/1024),
            MCSEntry(6, 2, 449, 2*449/1024),
            MCSEntry(7, 2, 526, 2*526/1024),
            MCSEntry(8, 2, 602, 2*602/1024),
            MCSEntry(9, 2, 679, 2*679/1024),
            MCSEntry(10, 4, 340, 4*340/1024),   # 16QAM
            MCSEntry(11, 4, 378, 4*378/1024),
            MCSEntry(12, 4, 434, 4*434/1024),
            MCSEntry(13, 4, 490, 4*490/1024),
            MCSEntry(14, 4, 553, 4*553/1024),
            MCSEntry(15, 4, 616, 4*616/1024),
            MCSEntry(16, 4, 658, 4*658/1024),
            MCSEntry(17, 6, 438, 6*438/1024),   # 64QAM
            MCSEntry(18, 6, 466, 6*466/1024),
            MCSEntry(19, 6, 517, 6*517/1024),
            MCSEntry(20, 6, 567, 6*567/1024),
            MCSEntry(21, 6, 616, 6*616/1024),
            MCSEntry(22, 6, 666, 6*666/1024),
            MCSEntry(23, 6, 719, 6*719/1024),
            MCSEntry(24, 6, 772, 6*772/1024),
            MCSEntry(25, 6, 822, 6*822/1024),
            MCSEntry(26, 6, 873, 6*873/1024),
            MCSEntry(27, 6, 910, 6*910/1024),
            MCSEntry(28, 6, 948, 6*948/1024)
        ]
    
    def calculate_tbs(self, mcs_index: int, direction: str) -> int:
        """
        Calculate Transport Block Size based on MCS and resource allocation.
        
        Implements simplified TBS calculation from 3GPP TS 38.214.
        
        Args:
            mcs_index: Modulation and Coding Scheme index
            direction: 'ul' or 'dl'
            
        Returns:
            Transport block size in bits
        """
        if mcs_index >= len(self.mcs_table):
            mcs_index = len(self.mcs_table) - 1
        
        mcs_entry = self.mcs_table[mcs_index]
        
        # Get resource parameters for direction
        if direction == 'ul':
            nrb = self.nrb_ul
            nsym_data = self.nsym_data_ul
        else:
            nrb = self.nrb_dl
            nsym_data = self.nsym_data_dl
        
        # Calculate effective REs (12 subcarriers per RB)
        nre_eff = nrb * nsym_data * 12
        
        # Calculate information bits
        n_info = nre_eff * mcs_entry.qm * mcs_entry.r / 1024
        
        # TBS calculation (simplified, minimum 24 bits)
        tbs_bits = max(int(n_info - 24), 24)
        
        # Align to 8 bits (byte alignment)
        tbs_bits = ((tbs_bits + 7) // 8) * 8
        
        return tbs_bits
    
    def calculate_tb_transmission_rate(self, app_type: str, direction: str) -> float:
        """
        Calculate TB transmission rate (TBs per second) for application.
        
        Args:
            app_type: Application type from configuration
            direction: 'ul' or 'dl'
            
        Returns:
            TB transmission rate in Hz (TBs per second)
        """
        app_config = self.config['applications'][app_type]
        
        # Get data rate and duty cycle for direction
        if direction == 'ul':
            data_rate_kbps = app_config['ul_data_rate_kbps']
            duty_cycle = app_config['duty_cycle_ul']
            mcs_index = self.default_mcs_ul
        else:
            data_rate_kbps = app_config['dl_data_rate_kbps']
            duty_cycle = app_config['duty_cycle_dl']
            mcs_index = self.default_mcs_dl
        
        # Calculate TBS for this direction
        tbs_bits = self.calculate_tbs(mcs_index, direction)
        
        # Calculate TB transmission rate
        if self.rates_are_instantaneous:
            instantaneous_rate_bps = data_rate_kbps * 1000
            avg_rate_bps = instantaneous_rate_bps * duty_cycle
        else:
            avg_rate_bps = data_rate_kbps * 1000
        
        # TB transmission rate = data rate / TBS size
        if tbs_bits > 0:
            tb_rate_hz = avg_rate_bps / tbs_bits
        else:
            tb_rate_hz = 0.0
        
        return tb_rate_hz
    
    def _dbm_to_linear_mw(self, power_dbm: float) -> float:
        """Convert power from dBm to linear mW"""
        return 10 ** (power_dbm / 10)
    
    def calculate_required_tx_power_dual(self, access_type: AccessType, distance_3d_m: float,
                                       elevation_angle_deg: float, target_snr_db: float,
                                       analysis_mode: str = "power_constrained",
                                       direction: str = "uplink",
                                       app_type: str = "medium") -> PowerResult:
        """
        Calculate required UE transmit power with TB-level support.
        
        Args:
            access_type: Type of access network
            distance_3d_m: 3D distance in meters
            elevation_angle_deg: Elevation angle in degrees
            target_snr_db: Target SNR in dB
            analysis_mode: "power_constrained" or "snr_target"
            direction: "uplink" or "downlink"
            app_type: Application type for TB calculations
            
        Returns:
            PowerResult with transmit power and energy metrics
        """
        frequency_hz, bandwidth_hz = self.channel_models.get_frequency_and_bandwidth(access_type)
        path_loss_db = self.channel_models.calculate_path_loss(
            access_type, distance_3d_m, elevation_angle_deg, frequency_hz
        )
        noise_power_dbm = self.channel_models.calculate_noise_power(bandwidth_hz, self.noise_figure_db)
        
        # Get antenna gains
        if access_type == AccessType.TERRESTRIAL:
            bs_antenna_gain_dbi = self.config['terrestrial']['antenna_gain_dbi']
        elif access_type == AccessType.UAV:
            bs_antenna_gain_dbi = self.config['uav']['antenna_gain_dbi']
        elif access_type == AccessType.HAPS:
            bs_antenna_gain_dbi = self.config['haps']['antenna_gain_dbi']
        elif access_type == AccessType.SATELLITE:
            gt_ratio_db_k = self.config['satellite']['gt_ratio_db_k']
            bs_antenna_gain_dbi = gt_ratio_db_k + 10 * math.log10(290)
        else:
            bs_antenna_gain_dbi = 0.0
        
        required_rx_power_dbm = noise_power_dbm + target_snr_db
        
        # Calculate TB-level metrics
        mcs_index = self.default_mcs_ul if direction == "uplink" else self.default_mcs_dl
        tbs_bits = self.calculate_tbs(mcs_index, 'ul' if direction == "uplink" else 'dl')
        
        if direction == "downlink":
            # Downlink: BS transmits, UE receives
            bs_tx_power_dbm = self.config.get(access_type.value, {}).get('tx_power_dbm', 30.0)
            rx_power_dbm = (bs_tx_power_dbm - path_loss_db +
                          bs_antenna_gain_dbi + self.ue_antenna_gain_dbi)
            achieved_snr_db = rx_power_dbm - noise_power_dbm
            
            return PowerResult(
                tx_power_dbm=0.0,
                tx_power_linear_mw=0.0,
                feasible=True,
                snr_achieved_db=achieved_snr_db,
                analysis_mode=analysis_mode,
                energy_per_tb_uj=0.0,
                tbs_bits=tbs_bits,
                mcs_index=mcs_index
            )
        
        # Uplink analysis
        required_tx_power_dbm = (required_rx_power_dbm + path_loss_db -
                               self.ue_antenna_gain_dbi - bs_antenna_gain_dbi)
        
        if analysis_mode == "power_constrained":
            feasible = required_tx_power_dbm <= self.max_tx_power_dbm
            
            if feasible:
                tx_power_dbm = required_tx_power_dbm
                achieved_snr_db = target_snr_db
                power_deficit_db = 0.0
            else:
                tx_power_dbm = self.max_tx_power_dbm
                rx_power_dbm = (self.max_tx_power_dbm - path_loss_db +
                              self.ue_antenna_gain_dbi + bs_antenna_gain_dbi)
                achieved_snr_db = rx_power_dbm - noise_power_dbm
                power_deficit_db = required_tx_power_dbm - self.max_tx_power_dbm
            
            # Calculate energy per TB
            tx_power_linear_mw = self._dbm_to_linear_mw(tx_power_dbm)
            tb_transmission_time_ms = self.slot_duration_ms
            total_power_mw = self.circuit_power_mw + tx_power_linear_mw
            energy_per_tb_uj = total_power_mw * tb_transmission_time_ms
            
            return PowerResult(
                tx_power_dbm=tx_power_dbm,
                tx_power_linear_mw=tx_power_linear_mw,
                feasible=feasible,
                snr_achieved_db=achieved_snr_db,
                analysis_mode="power_constrained",
                required_power_dbm=required_tx_power_dbm,
                power_deficit_db=power_deficit_db,
                energy_per_tb_uj=energy_per_tb_uj,
                tbs_bits=tbs_bits,
                mcs_index=mcs_index
            )
        
        elif analysis_mode == "snr_target":
            feasible = required_tx_power_dbm <= self.max_tx_power_dbm
            tx_power_dbm = required_tx_power_dbm
            achieved_snr_db = target_snr_db
            power_deficit_db = max(0, required_tx_power_dbm - self.max_tx_power_dbm)
            
            # Calculate energy per TB
            tx_power_linear_mw = self._dbm_to_linear_mw(tx_power_dbm)
            tb_transmission_time_ms = self.slot_duration_ms
            total_power_mw = self.circuit_power_mw + tx_power_linear_mw
            energy_per_tb_uj = total_power_mw * tb_transmission_time_ms
            
            return PowerResult(
                tx_power_dbm=tx_power_dbm,
                tx_power_linear_mw=tx_power_linear_mw,
                feasible=feasible,
                snr_achieved_db=achieved_snr_db,
                analysis_mode="snr_target",
                required_power_dbm=required_tx_power_dbm,
                power_deficit_db=power_deficit_db,
                energy_per_tb_uj=energy_per_tb_uj,
                tbs_bits=tbs_bits,
                mcs_index=mcs_index
            )
        
        else:
            raise ValueError(f"Unknown analysis mode: {analysis_mode}")
    
    def analyze_access_energy_raw(self, access_type: AccessType, distance_3d_m: float,
                                elevation_angle_deg: float, app_type: str = "medium",
                                analysis_mode: str = "power_constrained") -> EnergyAnalysis:
        """
        Energy analysis using TB-level calculations only.
        
        Args:
            access_type: Type of access network
            distance_3d_m: 3D distance in meters
            elevation_angle_deg: Elevation angle in degrees
            app_type: Application type
            analysis_mode: "power_constrained" or "snr_target"
            
        Returns:
            EnergyAnalysis with TB-level power and energy metrics
        """
        target_snr_ul_db = self.channel_config['target_snr_db']['ul_data']
        target_snr_dl_db = self.channel_config['target_snr_db']['dl_data']
        
        ul_power = self.calculate_required_tx_power_dual(
            access_type, distance_3d_m, elevation_angle_deg, target_snr_ul_db,
            analysis_mode, "uplink", app_type
        )
        
        dl_power = self.calculate_required_tx_power_dual(
            access_type, distance_3d_m, elevation_angle_deg, target_snr_dl_db,
            analysis_mode, "downlink", app_type
        )
        
        # TB-level metrics
        avg_tbs_bits = (ul_power.tbs_bits + dl_power.tbs_bits) / 2
        total_energy_per_tb = ul_power.energy_per_tb_uj + dl_power.energy_per_tb_uj
        
        # Calculate TB transmission rates
        ul_tb_rate_hz = self.calculate_tb_transmission_rate(app_type, 'ul')
        dl_tb_rate_hz = self.calculate_tb_transmission_rate(app_type, 'dl')
        avg_tb_rate_hz = (ul_tb_rate_hz + dl_tb_rate_hz) / 2
        
        return EnergyAnalysis(
            ul_power=ul_power,
            dl_power=dl_power,
            energy_saving_ratio=0.0,
            lifetime_extension_factor=1.0,
            analysis_mode=analysis_mode,
            total_energy_per_tb_uj=total_energy_per_tb,
            avg_tbs_bits=avg_tbs_bits,
            tb_transmission_rate_hz=avg_tb_rate_hz
        )
