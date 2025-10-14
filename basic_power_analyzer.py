"""
Basic UE Power Analyzer with Monte Carlo Statistics

Analyzes basic UE power consumption across all access types with:
- Required transmit power analysis
- Energy per bit and energy per TB calculations
- Monte Carlo sampling for statistical analysis
- Comparison across terrestrial, UAV, HAPS, and satellite

Key metrics:
- Required UL transmit power (dBm)
- Energy per bit (μJ/bit)
- Energy per TB (μJ/TB)
- Power feasibility analysis
- Energy savings vs satellite baseline
"""

import numpy as np
import pandas as pd
import yaml
import datetime
from typing import Dict, List, Tuple, Optional

from constellation import ConstellationSimulator
from channel_models import ChannelModels, AccessType
from power_models import UEPowerModels
from geometry_utils import GeometryUtils


class BasicUEPowerAnalyzer:
    """
    Basic UE Power Analyzer with Monte Carlo statistics.
    
    Provides fundamental power consumption analysis across all access types,
    with statistical analysis via Monte Carlo sampling over satellite positions
    and channel realizations.
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
        
        # Results storage
        self.required_power_results = []
        self.energy_per_bit_results = []
        self.mc_results = None
        
        print("Basic UE Power Analyzer initialized")
    
    def analyze_required_transmit_power(self):
        """Analyze required transmit power from link budget"""
        best_satellite = self.constellation.get_serving_satellite(self.ue_position, "power_analysis_ue")
        all_geometries = self.geo_utils.get_all_access_geometries(self.ue_position, best_satellite)
        
        access_types = [AccessType.TERRESTRIAL, AccessType.UAV, AccessType.HAPS, AccessType.SATELLITE]
        
        channel_config = self.config['channel']
        target_snr_ul_db = channel_config['target_snr_db']['ul_data']
        target_snr_dl_db = channel_config['target_snr_db']['dl_data']
        
        for access_type in access_types:
            if access_type == AccessType.SATELLITE:
                if best_satellite is not None:
                    distance = best_satellite.distance_3d * 1000
                    elevation = best_satellite.elevation_angle
                else:
                    distance = 650000
                    elevation = 35
            elif access_type.value in all_geometries:
                distance = all_geometries[access_type.value]['distance_3d_m']
                elevation = all_geometries[access_type.value]['elevation_deg']
            else:
                continue
            
            ul_power_result = self.power_models.calculate_required_tx_power_dual(
                access_type, distance, elevation, target_snr_ul_db,
                analysis_mode="power_constrained", direction="uplink", app_type="medium"
            )
            
            ul_required_power_dbm = ul_power_result.required_power_dbm if ul_power_result.required_power_dbm else ul_power_result.tx_power_dbm
            
            self.required_power_results.append({
                'access_type': access_type.value,
                'distance_km': distance / 1000,
                'elevation_deg': elevation,
                'ul_required_power_dbm': ul_required_power_dbm,
                'ul_achievable_power_dbm': ul_power_result.tx_power_dbm,
                'ul_power_feasible': ul_power_result.feasible,
                'ul_power_deficit_db': ul_power_result.power_deficit_db if ul_power_result.power_deficit_db else 0.0,
                'ul_snr_achieved_db': ul_power_result.snr_achieved_db,
                'target_snr_ul_db': target_snr_ul_db
            })
    
    def analyze_energy_per_bit_tb_method(self):
        """Analyze energy per bit using Transport Block method"""
        best_satellite = self.constellation.get_serving_satellite(self.ue_position, "energy_bit_analysis_ue")
        all_geometries = self.geo_utils.get_all_access_geometries(self.ue_position, best_satellite)
        
        access_types = [AccessType.TERRESTRIAL, AccessType.UAV, AccessType.HAPS, AccessType.SATELLITE]
        
        self.energy_per_bit_results = []
        
        for access_type in access_types:
            if access_type == AccessType.SATELLITE:
                if best_satellite is not None:
                    distance = best_satellite.distance_3d * 1000
                    elevation = best_satellite.elevation_angle
                else:
                    distance = 650000
                    elevation = 35
            elif access_type.value in all_geometries:
                distance = all_geometries[access_type.value]['distance_3d_m']
                elevation = all_geometries[access_type.value]['elevation_deg']
            else:
                continue
            
            # Analyze using power_constrained mode
            energy_analysis = self.power_models.analyze_access_energy_raw(
                access_type, distance, elevation, "medium", "power_constrained"
            )
            
            # Extract metrics
            ul_energy_per_tb = energy_analysis.ul_power.energy_per_tb_uj
            dl_energy_per_tb = energy_analysis.dl_power.energy_per_tb_uj
            ul_tbs_bits = energy_analysis.ul_power.tbs_bits
            dl_tbs_bits = energy_analysis.dl_power.tbs_bits
            
            # Energy per bit (derived from TB-level)
            ul_energy_per_bit = energy_analysis.ul_power.energy_per_bit_uj
            dl_energy_per_bit = energy_analysis.dl_power.energy_per_bit_uj
            
            self.energy_per_bit_results.append({
                'access_type': access_type.value,
                'distance_km': distance / 1000,
                'elevation_deg': elevation,
                'ul_energy_per_bit_uj': ul_energy_per_bit,
                'dl_energy_per_bit_uj': dl_energy_per_bit,
                'ul_energy_per_tb_uj': ul_energy_per_tb,
                'dl_energy_per_tb_uj': dl_energy_per_tb,
                'ul_tbs_bits': ul_tbs_bits,
                'dl_tbs_bits': dl_tbs_bits,
                'ul_feasible': energy_analysis.ul_power.feasible,
                'total_energy_per_bit_uj': energy_analysis.total_energy_per_bit_uj,
                'total_energy_per_tb_uj': energy_analysis.total_energy_per_tb_uj
            })
    
    def run_monte_carlo_analysis(self, n_iterations: int = 100):
        """
        Run Monte Carlo simulation with instant time jumps.
        
        Args:
            n_iterations: Number of Monte Carlo iterations
            
        Returns:
            Dictionary with Monte Carlo results
        """
        print(f"\nMonte Carlo Analysis: {n_iterations} iterations")
        
        mc_required_power = []
        mc_energy_per_bit = []
        
        # Store original time
        with self.constellation.time_lock:
            original_time = self.constellation.current_time
        
        # Calculate time step to sample across full orbital pass
        orbital_period_sec = self.constellation.orbital_period
        time_step_sec = orbital_period_sec / n_iterations
        
        for i in range(n_iterations):
            if (i + 1) % 10 == 0:
                print(f"Running iteration {i+1}/{n_iterations}...")
            
            # Instant time jump
            with self.constellation.time_lock:
                self.constellation.current_time = original_time + datetime.timedelta(
                    seconds=i * time_step_sec
                )
            
            self.required_power_results = []
            self.energy_per_bit_results = []
            
            self.analyze_required_transmit_power()
            self.analyze_energy_per_bit_tb_method()
            
            if self.required_power_results:
                mc_required_power.append(pd.DataFrame(self.required_power_results))
            if self.energy_per_bit_results:
                mc_energy_per_bit.append(pd.DataFrame(self.energy_per_bit_results))
        
        # Restore original time
        with self.constellation.time_lock:
            self.constellation.current_time = original_time
        
        return {
            'required_power': mc_required_power,
            'energy_per_bit': mc_energy_per_bit,
            'n_iterations': n_iterations
        }
    
    def calculate_monte_carlo_statistics(self):
        """Calculate statistics from stored MC results"""
        if not hasattr(self, 'mc_results') or self.mc_results is None:
            print("No MC results available. Run run_complete_analysis() first.")
            return
        
        all_energy_dfs = self.mc_results['energy_per_bit']
        all_power_dfs = self.mc_results['required_power']
        
        combined_energy_df = pd.concat(all_energy_dfs, ignore_index=True)
        combined_power_df = pd.concat(all_power_dfs, ignore_index=True)
        
        print("\n" + "="*80)
        print(f"MONTE CARLO STATISTICS ({self.mc_results['n_iterations']} iterations)")
        print("="*80)
        
        # Required transmit power
        print("\n--- REQUIRED UL TRANSMIT POWER (dBm) ---")
        print(f"{'Access':<12} {'Median':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Feasible'}")
        print("-"*75)
        
        for access_type in ['terrestrial', 'uav', 'haps', 'satellite']:
            data = combined_power_df[combined_power_df['access_type'] == access_type]
            powers = data['ul_required_power_dbm']
            
            median_pwr = powers.median()
            mean_pwr = powers.mean()
            std_pwr = powers.std()
            min_pwr = powers.min()
            max_pwr = powers.max()
            feasible_pct = (data['ul_power_feasible'].sum() / len(data)) * 100
            
            feasible_str = f"{feasible_pct:.0f}%" if feasible_pct < 100 else "Yes"
            print(f"{access_type:<12} {median_pwr:<10.2f} {mean_pwr:<10.2f} {std_pwr:<10.2f} {min_pwr:<10.2f} {max_pwr:<10.2f} {feasible_str}")
        
        # Power gap analysis
        print("\n--- POWER GAP ANALYSIS ---")
        terr_data = combined_power_df[combined_power_df['access_type'] == 'terrestrial']
        sat_data_pwr = combined_power_df[combined_power_df['access_type'] == 'satellite']
        
        if not terr_data.empty and not sat_data_pwr.empty:
            terr_median = terr_data['ul_required_power_dbm'].median()
            sat_median = sat_data_pwr['ul_required_power_dbm'].median()
            power_gap = sat_median - terr_median
            ue_max = self.ue_config['max_tx_power_dbm']
            sat_excess = sat_median - ue_max
            
            print(f"Terrestrial median: {terr_median:.2f} dBm")
            print(f"Satellite median: {sat_median:.2f} dBm")
            print(f"Power gap (Sat - Terr): {power_gap:.1f} dB")
            print(f"Satellite exceeds UE limit (23 dBm) by: {sat_excess:.1f} dB")
        
        # Energy per bit
        print("\n--- ENERGY PER BIT (μJ/bit) ---")
        print(f"{'Access':<12} {'UL Mean':<12} {'UL Std':<10} {'DL Mean':<12} {'DL Std':<10}")
        print("-"*60)
        
        for access_type in ['terrestrial', 'uav', 'haps', 'satellite']:
            data = combined_energy_df[combined_energy_df['access_type'] == access_type]
            ul_mean = data['ul_energy_per_bit_uj'].mean()
            ul_std = data['ul_energy_per_bit_uj'].std()
            dl_mean = data['dl_energy_per_bit_uj'].mean()
            dl_std = data['dl_energy_per_bit_uj'].std()
            
            print(f"{access_type:<12} {ul_mean:<12.4f} {ul_std:<10.4f} {dl_mean:<12.4f} {dl_std:<10.4f}")
        
        # Energy savings (Total Energy)
        print("\n--- ENERGY SAVINGS vs SATELLITE (Total Energy: UL + DL) ---")
        
        sat_data_energy = combined_energy_df[combined_energy_df['access_type'] == 'satellite']
        sat_total_energies = sat_data_energy['ul_energy_per_bit_uj'] + sat_data_energy['dl_energy_per_bit_uj']
        sat_total_mean = sat_total_energies.mean()
        
        print(f"{'Access':<12} {'Total Energy':<15} {'Savings %':<15} {'Lifetime Factor':<15}")
        print("-"*60)
        
        for access_type in ['terrestrial', 'uav', 'haps']:
            data = combined_energy_df[combined_energy_df['access_type'] == access_type]
            total_energies = data['ul_energy_per_bit_uj'] + data['dl_energy_per_bit_uj']
            total_mean = total_energies.mean()
            
            savings = ((sat_total_mean - total_mean) / sat_total_mean) * 100
            lifetime = sat_total_mean / total_mean
            
            print(f"{access_type:<12} {total_mean:<15.4f} {savings:<15.2f} {lifetime:<15.2f}x")
        
        print(f"{'satellite':<12} {sat_total_mean:<15.4f} {'--':<15} {'1.00x':<15}")
        
        print("\n" + "="*80)
    
    def run_complete_analysis(self, monte_carlo_iterations: int = 100):
        """
        Run complete analysis with Monte Carlo.
        
        Args:
            monte_carlo_iterations: Number of Monte Carlo iterations
        """
        print("=" * 80)
        print(f"BASIC UE POWER ANALYSIS (Monte Carlo n={monte_carlo_iterations})")
        print("=" * 80)
        
        # Run Monte Carlo and store results
        self.mc_results = self.run_monte_carlo_analysis(monte_carlo_iterations)
        
        # Calculate statistics
        self.calculate_monte_carlo_statistics()
        
        print("\n" + "=" * 80)
        print("Analysis completed!")
        print("=" * 80)


if __name__ == "__main__":
    try:
        analyzer = BasicUEPowerAnalyzer("config/parameters.yaml")
        analyzer.run_complete_analysis(monte_carlo_iterations=100)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
