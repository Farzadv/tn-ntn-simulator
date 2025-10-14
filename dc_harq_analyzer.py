"""
UE Activity Factor HARQ Power Analysis with Monte Carlo Averaging

Analyzes HARQ power consumption across different UE activity factors with:
- Monte Carlo averaging over random shadowing for smooth results
- TB-level power calculations
- HARQ feedback energy analysis
- Bundling strategies (Independent vs Bundled)

Key metrics:
- Data transmission power (UL/DL)
- HARQ feedback power (UL/DL)
- Total power consumption
- Power reduction with bundling
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from constellation import ConstellationSimulator
from channel_models import ChannelModels, AccessType
from power_models import UEPowerModels
from geometry_utils import GeometryUtils


@dataclass
class ActivityFactorResult:
    """Result for one UE activity factor configuration"""
    harq_strategy: str
    total_activity: float
    ul_activity: float
    dl_activity: float
    bundle_size: int
    
    # Power components (mW) - separated UL/DL
    ul_data_power_mw: float
    dl_data_power_mw: float
    ul_harq_power_mw: float
    dl_harq_power_mw: float
    total_power_mw: float
    
    # Additional info
    ul_power_dbm: float
    dl_power_dbm: float
    
    # Satellite tracking
    satellite_id: Optional[int] = None
    satellite_distance_km: Optional[float] = None
    satellite_elevation_deg: Optional[float] = None


class UEActivityHARQAnalyzer:
    """
    Analyzes HARQ power consumption across UE activity factor variations.
    
    Uses TB-level power calculations with Monte Carlo averaging over shadowing
    to provide smooth, realistic power consumption estimates.
    """
    
    def __init__(self, config_path: str = "config/parameters.yaml"):
        """
        Initialize analyzer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize simulator modules
        self.constellation = ConstellationSimulator(config_path)
        self.channel_models = ChannelModels(config_path)
        self.power_models = UEPowerModels(config_path, self.constellation)
        self.geo_utils = GeometryUtils(config_path)
        
        # UE configuration
        self.ue_config = self.config['ue']
        self.ue_position = tuple(self.ue_config['location'])
        
        # Analysis parameters
        self.app_type = "medium"
        self.activity_values = sorted([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                                      0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        self.bundle_sizes = [2, 4, 6, 8, 10, 12, 14]
        self.ul_dl_ratios = [
            (0.10, 0.90, "10UL-90DL"),
            (0.50, 0.50, "50UL-50DL")
        ]
        
        # Monte Carlo averaging configuration
        self.n_mc_samples = 100
        
        # Get 5G NR timing parameters
        nr_config = self.config.get('5g_nr', {})
        self.slot_duration_ms = 0.5
        if 'slot_duration_ms' in nr_config:
            self.slot_duration_ms = nr_config['slot_duration_ms']
        
        # HARQ configuration
        harq_config = self.config.get('harq', {})
        self.max_retransmissions = harq_config.get('max_retransmissions', 4)
        
        # DUDe HARQ configuration
        dude_config = self.config.get('dude_harq', {})
        self.ack_nack_processing_time_ms = dude_config.get('ack_nack_processing_time_ms', 1.2)
        self.harq_feedback_preparation_time_ms = dude_config.get('harq_feedback_preparation_time_ms', 0.8)
        
        self.all_results = {}
        
        print("UE Activity Factor HARQ Power Analyzer initialized (MONTE CARLO AVERAGED)")
        print(f"  Application: {self.app_type}")
        print(f"  Activity factor range: {min(self.activity_values)*100:.0f}% - {max(self.activity_values)*100:.0f}%")
        print(f"  Bundle sizes: {self.bundle_sizes}")
        print(f"  UL:DL ratios: {[r[2] for r in self.ul_dl_ratios]}")
        print(f"  Monte Carlo samples: {self.n_mc_samples}")
    
    def calculate_data_power_averaged(self, ul_activity: float, dl_activity: float,
                                     n_samples: int = 50) -> Tuple[float, float]:
        """
        Calculate data transmission power with Monte Carlo averaging over shadowing.
        
        Args:
            ul_activity: UL activity factor
            dl_activity: DL activity factor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            (ul_avg_power_mw, dl_avg_power_mw)
        """
        ul_powers = []
        dl_powers = []
        
        for _ in range(n_samples):
            # Get geometry (shadowing varies each call)
            best_satellite = self.constellation.get_serving_satellite(
                self.ue_position, "activity_harq_ue"
            )
            all_geometries = self.geo_utils.get_all_access_geometries(
                self.ue_position, best_satellite
            )
            
            distance = all_geometries['terrestrial']['distance_3d_m']
            elevation = all_geometries['terrestrial']['elevation_deg']
            
            # Get target SNR
            target_snr_ul = self.config['channel']['target_snr_db']['ul_data']
            
            # Calculate required UL TX power
            ul_power_result = self.power_models.calculate_required_tx_power_dual(
                AccessType.TERRESTRIAL, distance, elevation, target_snr_ul,
                analysis_mode="power_constrained", direction="uplink", app_type=self.app_type
            )
            
            ul_tx_power_mw = ul_power_result.tx_power_linear_mw
            circuit_power_mw = self.ue_config['circuit_power_mw']
            rx_power_mw = self.ue_config['rx_power_mw']
            
            # Apply activity factors
            ul_avg_power_mw = (ul_tx_power_mw + circuit_power_mw) * ul_activity
            dl_avg_power_mw = (rx_power_mw + circuit_power_mw) * dl_activity
            
            ul_powers.append(ul_avg_power_mw)
            dl_powers.append(dl_avg_power_mw)
        
        # Return averaged values
        return (np.mean(ul_powers), np.mean(dl_powers))
    
    def calculate_harq_power_averaged(self, harq_strategy: str, ul_activity: float,
                                     dl_activity: float, bundle_size: int = 1,
                                     n_samples: int = 50) -> Tuple[float, float, Optional[int], Optional[float], Optional[float]]:
        """
        Calculate HARQ feedback power with Monte Carlo averaging.
        
        Args:
            harq_strategy: "harq_less", "independent", or "bundled"
            ul_activity: UL activity factor
            dl_activity: DL activity factor
            bundle_size: Bundle size for bundled strategy
            n_samples: Number of Monte Carlo samples
            
        Returns:
            (ul_harq_power_mw, dl_harq_power_mw, sat_id, sat_distance_km, sat_elevation_deg)
        """
        ul_harq_powers = []
        dl_harq_powers = []
        last_sat_info = (None, None, None)
        
        for _ in range(n_samples):
            # Get application data rates
            app_config = self.config['applications'][self.app_type]
            ul_rate_kbps = app_config['ul_data_rate_kbps']
            dl_rate_kbps = app_config['dl_data_rate_kbps']
            
            # Calculate Transport Block sizes
            mcs_ul = self.power_models.default_mcs_ul
            mcs_dl = self.power_models.default_mcs_dl
            tbs_ul = self.power_models.calculate_tbs(mcs_ul, 'ul')
            tbs_dl = self.power_models.calculate_tbs(mcs_dl, 'dl')
            
            # Calculate TB transmission rates
            ul_tb_rate = (ul_rate_kbps * 1000 * ul_activity) / tbs_ul if tbs_ul > 0 else 0
            dl_tb_rate = (dl_rate_kbps * 1000 * dl_activity) / tbs_dl if tbs_dl > 0 else 0
            
            # UL HARQ: Receive ACK/NACK from TERRESTRIAL
            ul_harq_energy_per_feedback = (
                self.ue_config['circuit_power_mw'] +
                self.ue_config['rx_power_mw']
            ) * self.ack_nack_processing_time_ms
            
            ul_harq_rate = ul_tb_rate
            ul_harq_power_mw = (ul_harq_energy_per_feedback * ul_harq_rate) / 1000
            
            # DL HARQ: Transmit ACK/NACK to SATELLITE
            if harq_strategy == "harq_less":
                dl_harq_power_mw = 0.0
                ul_harq_powers.append(ul_harq_power_mw)
                dl_harq_powers.append(dl_harq_power_mw)
                continue
            
            # Get satellite geometry
            best_satellite = self.constellation.get_serving_satellite(
                self.ue_position, "dl_harq_power_calc"
            )
            
            if best_satellite:
                sat_distance_m = best_satellite.distance_3d * 1000
                sat_elevation = best_satellite.elevation_angle
                sat_id = best_satellite.sat_id
                last_sat_info = (sat_id, sat_distance_m/1000, sat_elevation)
            else:
                sat_distance_m = 600000
                sat_elevation = 35
                sat_id = None
            
            # Calculate required TX power (SNR-TARGET MODE)
            target_snr_ul = self.config['channel']['target_snr_db']['ul_data']
            
            try:
                sat_power_result = self.power_models.calculate_required_tx_power_dual(
                    AccessType.SATELLITE,
                    sat_distance_m,
                    sat_elevation,
                    target_snr_ul,
                    analysis_mode="snr_target",
                    direction="uplink",
                    app_type=self.app_type
                )
                
                dl_harq_tx_power_mw = sat_power_result.tx_power_linear_mw
            
            except Exception as e:
                dl_harq_tx_power_mw = 10 ** (25 / 10)
            
            # Calculate ACK/NACK message size scaled by bundle size
            dude_config = self.config.get('dude_harq', {})
            base_ack_nack_bits = dude_config.get('ack_nack_message_bits', 8)
            feedback_bits_per_tb = dude_config.get('feedback_bits_per_tb', 2)
            
            if harq_strategy == "bundled":
                ack_nack_bits = base_ack_nack_bits + (bundle_size - 1) * feedback_bits_per_tb
            else:
                ack_nack_bits = base_ack_nack_bits
            
            overhead_factor = dude_config.get('transmission_overhead_factor', 1.2)
            
            # Calculate transmission time
            transmission_time_ms = (ack_nack_bits / (ul_rate_kbps * 1000)) * 1000 * overhead_factor
            transmission_time_ms = max(transmission_time_ms, 0.1)
            
            # Separate phases
            preparation_energy_uj = (
                self.ue_config['circuit_power_mw'] * self.harq_feedback_preparation_time_ms
            )
            
            transmission_energy_uj = (
                self.ue_config['circuit_power_mw'] + dl_harq_tx_power_mw
            ) * transmission_time_ms
            
            dl_harq_energy_per_feedback = preparation_energy_uj + transmission_energy_uj
            
            # Apply strategy
            if harq_strategy == "independent":
                dl_harq_rate = dl_tb_rate
            elif harq_strategy == "bundled":
                dl_harq_rate = dl_tb_rate / bundle_size
            else:
                dl_harq_rate = 0
            
            dl_harq_power_mw = (dl_harq_energy_per_feedback * dl_harq_rate) / 1000
            
            ul_harq_powers.append(ul_harq_power_mw)
            dl_harq_powers.append(dl_harq_power_mw)
        
        # Return averaged values and last satellite info
        return (np.mean(ul_harq_powers), np.mean(dl_harq_powers),
                last_sat_info[0], last_sat_info[1], last_sat_info[2])
    
    def analyze_activity_variation(self, ul_ratio: float, dl_ratio: float) -> List[ActivityFactorResult]:
        """
        Analyze all activity factor values for given UL:DL ratio.
        
        Args:
            ul_ratio: Uplink activity ratio
            dl_ratio: Downlink activity ratio
            
        Returns:
            List of ActivityFactorResult objects
        """
        results = []
        
        print(f"\nAnalyzing UL:DL ratio {ul_ratio*100:.0f}:{dl_ratio*100:.0f}")
        print(f"Monte Carlo averaging: {self.n_mc_samples} samples per activity factor")
        print("-" * 80)
        
        total_iterations = len(self.activity_values)
        
        for idx, total_activity in enumerate(self.activity_values):
            ul_activity = total_activity * ul_ratio
            dl_activity = total_activity * dl_ratio
            
            # Progress indicator
            progress = ((idx + 1) / total_iterations) * 100
            print(f"  Processing activity {total_activity*100:5.1f}% [{idx+1}/{total_iterations}] "
                  f"({progress:5.1f}% complete)...", end='\r')
            
            # Use averaged calculations
            ul_data_power, dl_data_power = self.calculate_data_power_averaged(
                ul_activity, dl_activity, self.n_mc_samples
            )
            
            # HARQ-less
            ul_harq_power, dl_harq_power, sat_id, sat_dist, sat_elev = \
                self.calculate_harq_power_averaged("harq_less", ul_activity, dl_activity,
                                                  1, self.n_mc_samples)
            results.append(ActivityFactorResult(
                harq_strategy="harq_less",
                total_activity=total_activity,
                ul_activity=ul_activity,
                dl_activity=dl_activity,
                bundle_size=1,
                ul_data_power_mw=ul_data_power,
                dl_data_power_mw=dl_data_power,
                ul_harq_power_mw=ul_harq_power,
                dl_harq_power_mw=dl_harq_power,
                total_power_mw=ul_data_power + dl_data_power + ul_harq_power + dl_harq_power,
                ul_power_dbm=0,
                dl_power_dbm=0,
                satellite_id=sat_id,
                satellite_distance_km=sat_dist,
                satellite_elevation_deg=sat_elev
            ))
            
            # Independent
            ul_harq_power, dl_harq_power, sat_id, sat_dist, sat_elev = \
                self.calculate_harq_power_averaged("independent", ul_activity, dl_activity,
                                                  1, self.n_mc_samples)
            results.append(ActivityFactorResult(
                harq_strategy="independent",
                total_activity=total_activity,
                ul_activity=ul_activity,
                dl_activity=dl_activity,
                bundle_size=1,
                ul_data_power_mw=ul_data_power,
                dl_data_power_mw=dl_data_power,
                ul_harq_power_mw=ul_harq_power,
                dl_harq_power_mw=dl_harq_power,
                total_power_mw=ul_data_power + dl_data_power + ul_harq_power + dl_harq_power,
                ul_power_dbm=0,
                dl_power_dbm=0,
                satellite_id=sat_id,
                satellite_distance_km=sat_dist,
                satellite_elevation_deg=sat_elev
            ))
            
            # Bundled with different bundle sizes
            for bundle_size in self.bundle_sizes:
                ul_harq_power, dl_harq_power, sat_id, sat_dist, sat_elev = \
                    self.calculate_harq_power_averaged("bundled", ul_activity, dl_activity,
                                                      bundle_size, self.n_mc_samples)
                results.append(ActivityFactorResult(
                    harq_strategy=f"bundled_{bundle_size}",
                    total_activity=total_activity,
                    ul_activity=ul_activity,
                    dl_activity=dl_activity,
                    bundle_size=bundle_size,
                    ul_data_power_mw=ul_data_power,
                    dl_data_power_mw=dl_data_power,
                    ul_harq_power_mw=ul_harq_power,
                    dl_harq_power_mw=dl_harq_power,
                    total_power_mw=ul_data_power + dl_data_power + ul_harq_power + dl_harq_power,
                    ul_power_dbm=0,
                    dl_power_dbm=0,
                    satellite_id=sat_id,
                    satellite_distance_km=sat_dist,
                    satellite_elevation_deg=sat_elev
                ))
        
        print()  # New line after progress indicator
        print(f"\n  Generated {len(results)} results (averaged over {self.n_mc_samples} samples each)")
        print("-" * 80)
        
        return results
    
    def print_statistics(self, ratio_name: str):
        """
        Print statistics for a specific UL:DL ratio.
        
        Args:
            ratio_name: Name of the ratio (e.g., "10UL-90DL")
        """
        if ratio_name not in self.all_results:
            print(f"No results for {ratio_name}")
            return
        
        results = self.all_results[ratio_name]
        
        print(f"\n{'='*80}")
        print(f"STATISTICS FOR {ratio_name}")
        print(f"{'='*80}")
        
        # Get representative activity point (50%)
        target_activity = 0.50
        
        print(f"\nPower Breakdown at {target_activity*100:.0f}% Activity Factor:")
        print(f"{'Strategy':<20} {'UL Data':<12} {'DL Data':<12} {'UL HARQ':<12} {'DL HARQ':<12} {'Total':<12}")
        print("-"*90)
        
        for strategy_key in ['harq_less', 'independent', 'bundled_2']:
            strategy_results = [r for r in results
                              if r.harq_strategy == strategy_key
                              and abs(r.total_activity - target_activity) < 0.01]
            
            if strategy_results:
                result = strategy_results[0]
                print(f"{result.harq_strategy:<20} {result.ul_data_power_mw:>8.1f} mW  "
                      f"{result.dl_data_power_mw:>8.1f} mW  {result.ul_harq_power_mw:>8.1f} mW  "
                      f"{result.dl_harq_power_mw:>8.1f} mW  {result.total_power_mw:>8.1f} mW")
        
        # Power reduction analysis
        print(f"\nPower Reduction with Bundling (at {target_activity*100:.0f}% activity):")
        
        # Get Independent baseline
        independent_results = [r for r in results
                             if r.harq_strategy == 'independent'
                             and abs(r.total_activity - target_activity) < 0.01]
        
        if independent_results:
            baseline_power = independent_results[0].total_power_mw
            
            print(f"  Independent (baseline): {baseline_power:.1f} mW")
            
            for bundle_size in [2, 4, 6, 8]:
                bundled_results = [r for r in results
                                  if r.harq_strategy == f'bundled_{bundle_size}'
                                  and abs(r.total_activity - target_activity) < 0.01]
                
                if bundled_results:
                    bundled_power = bundled_results[0].total_power_mw
                    reduction_pct = ((baseline_power - bundled_power) / baseline_power) * 100
                    
                    print(f"  Bundled (size={bundle_size}): {bundled_power:.1f} mW "
                          f"({reduction_pct:.1f}% reduction)")
        
        print(f"{'='*80}")
    
    def run_complete_analysis(self):
        """Run complete analysis with Monte Carlo averaging"""
        print("=" * 80)
        print("UE ACTIVITY FACTOR HARQ POWER ANALYSIS (MONTE CARLO AVERAGED)")
        print("=" * 80)
        print(f"Application: {self.app_type}")
        print(f"Activity factor range: {[f'{a*100:.0f}%' for a in [min(self.activity_values), max(self.activity_values)]]}")
        print(f"Bundle sizes: {self.bundle_sizes}")
        print(f"Monte Carlo samples per point: {self.n_mc_samples}")
        print("=" * 80)
        
        for ul_ratio, dl_ratio, ratio_name in self.ul_dl_ratios:
            print(f"\n{'='*80}")
            print(f"Processing {ratio_name}")
            print(f"{'='*80}")
            
            results = self.analyze_activity_variation(ul_ratio, dl_ratio)
            self.all_results[ratio_name] = results
            print(f"  ✓ Completed {ratio_name}: {len(results)} results")
        
        # Print statistics for each ratio
        for ul_ratio, dl_ratio, ratio_name in self.ul_dl_ratios:
            self.print_statistics(ratio_name)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results are Monte Carlo averaged over {self.n_mc_samples} samples")
        print("=" * 80)


if __name__ == "__main__":
    try:
        analyzer = UEActivityHARQAnalyzer("config/parameters.yaml")
        analyzer.run_complete_analysis()
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
