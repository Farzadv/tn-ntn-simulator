"""
BLER-Focused HARQ Strategy Analyzer

Analyzes Block Error Rate (BLER) performance across different HARQ strategies:
- HARQ-less: No retransmissions
- Independent: Separate HARQ processes per TB
- Bundled: Aggregated HARQ feedback

Uses logistic BLER model and Chase Combining for retransmission analysis.
"""

import numpy as np
import yaml
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

from constellation import ConstellationSimulator
from channel_models import ChannelModels, AccessType
from power_models import UEPowerModels
from geometry_utils import GeometryUtils


class HARQStrategy(Enum):
    """HARQ strategies for analysis"""
    HARQ_LESS = "harq_less"
    INDEPENDENT = "independent"
    BUNDLED = "bundled"


@dataclass
class BLERResult:
    """BLER analysis result for one strategy"""
    harq_strategy: HARQStrategy
    app_type: str
    max_retx: int
    direction: str  # "ul", "dl", or "both"
    
    # BLER metrics (averaged over iterations)
    ul_bler_initial: float = 0.0
    ul_bler_final: float = 0.0
    dl_bler_initial: float = 0.0
    dl_bler_final: float = 0.0
    average_bler: float = 0.0
    
    # Standard deviations
    ul_bler_std: float = 0.0
    dl_bler_std: float = 0.0
    
    # Retransmission metrics
    expected_ul_retx: float = 0.0
    expected_dl_retx: float = 0.0
    
    # SNR info
    ul_snr_db: float = 0.0
    dl_snr_db: float = 0.0
    
    # MCS info
    mcs_ul: int = 0
    mcs_dl: int = 0


class BLERHARQAnalyzer:
    """
    BLER-focused HARQ Strategy Analyzer.
    
    Analyzes BLER performance across different HARQ strategies using:
    - Logistic BLER model
    - Chase Combining for retransmissions
    - Monte Carlo averaging over channel realizations
    """
    
    def __init__(self, config_path: str = "config/parameters.yaml"):
        """
        Initialize BLER analyzer with configuration.
        
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
        
        # HARQ configuration - sweep from 1 to 8
        self.max_retransmissions_list = list(range(1, 9))
        
        # Monte Carlo iterations
        self.mc_iterations = 30
        
        # BLER model parameters
        self.bler_model = self.config.get('bler_model', {})
        
        # Get MCS indices from config
        nr_config = self.config.get('5g_nr', {})
        self.mcs_ul = nr_config.get('mcs_ul', 10)
        self.mcs_dl = nr_config.get('mcs_dl', 15)
        
        # Results storage
        self.results: List[BLERResult] = []
        
        print("BLER HARQ Strategy Analyzer initialized")
        print(f"  Monte Carlo iterations: {self.mc_iterations}")
        print(f"  Max retransmissions sweep: {self.max_retransmissions_list}")
    
    def calculate_bler_logistic(self, snr_db: float, access_type: AccessType) -> float:
        """
        Calculate initial BLER using logistic model.
        
        BLER = 1 / (1 + exp(a*(SNR - SNR_0)))
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            access_type: Type of access for model parameters
            
        Returns:
            BLER value between 0 and 1
        """
        access_key = access_type.value if access_type.value in self.bler_model else 'terrestrial'
        params = self.bler_model[access_key]
        
        a = params['a']
        snr_0 = params['snr_0']
        
        # Logistic model
        bler = 1.0 / (1.0 + np.exp(a * (snr_db - snr_0)))
        
        return max(0.0001, min(0.5, bler))
    
    def calculate_bler_with_chase_combining(self, initial_bler: float,
                                           snr_db: float,
                                           access_type: AccessType,
                                           harq_strategy: HARQStrategy,
                                           direction: str,
                                           max_retx: int) -> Tuple[float, float]:
        """
        Calculate final BLER after Chase Combining retransmissions.
        
        Chase Combining improves SNR with each retransmission:
        Effective SNR = original SNR + 10*log10(number of transmissions)
        
        Args:
            initial_bler: Initial BLER before retransmissions
            snr_db: Original SNR in dB
            access_type: Type of access
            harq_strategy: HARQ strategy being used
            direction: "ul" or "dl"
            max_retx: Maximum number of retransmissions
            
        Returns:
            Tuple of (final_bler, expected_retransmissions)
        """
        # HARQ-less strategy: only affects DL, UL still has HARQ
        if harq_strategy == HARQStrategy.HARQ_LESS and direction == "dl":
            return initial_bler, 0.0
        
        # Chase Combining: SNR improves with number of transmissions
        # Total transmissions = 1 (original) + max_retx
        total_transmissions = 1 + max_retx
        snr_gain_db = 10 * math.log10(total_transmissions)
        effective_snr = snr_db + snr_gain_db
        
        # Calculate BLER with improved SNR
        final_bler = self.calculate_bler_logistic(effective_snr, access_type)
        
        # Expected retransmissions (if BLER is low, fewer retransmissions needed)
        expected_retx = initial_bler * max_retx
        
        return final_bler, expected_retx
    
    def analyze_bler_single_iteration(self, harq_strategy: HARQStrategy,
                                     app_type: str,
                                     max_retx: int) -> Dict:
        """
        Single iteration of BLER analysis.
        
        Args:
            harq_strategy: HARQ strategy to analyze
            app_type: Application type
            max_retx: Maximum retransmissions
            
        Returns:
            Dictionary with BLER metrics
        """
        # Get geometries
        best_satellite = self.constellation.get_serving_satellite(
            self.ue_position, f"bler_analysis_{app_type}"
        )
        all_geometries = self.geo_utils.get_all_access_geometries(
            self.ue_position, best_satellite
        )
        
        # UL: Terrestrial (short-range anchor)
        ul_distance = all_geometries['terrestrial']['distance_3d_m']
        ul_elevation = all_geometries['terrestrial']['elevation_deg']
        
        # DL: Satellite
        if best_satellite:
            dl_distance = best_satellite.distance_3d * 1000
            dl_elevation = best_satellite.elevation_angle
        else:
            dl_distance = 650000
            dl_elevation = 35
        
        # Get target SNRs
        target_snr_ul = self.config['channel']['target_snr_db']['ul_data']
        target_snr_dl = self.config['channel']['target_snr_db']['dl_data']
        
        # Calculate achieved SNRs
        try:
            ul_analysis = self.power_models.analyze_access_energy_raw(
                AccessType.TERRESTRIAL, ul_distance, ul_elevation,
                app_type, "power_constrained"
            )
            ul_snr = ul_analysis.ul_power.snr_achieved_db
        except:
            ul_snr = target_snr_ul
        
        try:
            dl_analysis = self.power_models.analyze_access_energy_raw(
                AccessType.SATELLITE, dl_distance, dl_elevation,
                app_type, "power_constrained"
            )
            dl_snr = dl_analysis.dl_power.snr_achieved_db
        except:
            dl_snr = target_snr_dl
        
        # Calculate initial BLER
        ul_bler_initial = self.calculate_bler_logistic(ul_snr, AccessType.TERRESTRIAL)
        dl_bler_initial = self.calculate_bler_logistic(dl_snr, AccessType.SATELLITE)
        
        # Calculate final BLER (after Chase Combining)
        ul_bler_final, ul_expected_retx = self.calculate_bler_with_chase_combining(
            ul_bler_initial, ul_snr, AccessType.TERRESTRIAL, harq_strategy, "ul", max_retx
        )
        dl_bler_final, dl_expected_retx = self.calculate_bler_with_chase_combining(
            dl_bler_initial, dl_snr, AccessType.SATELLITE, harq_strategy, "dl", max_retx
        )
        
        return {
            'ul_bler_initial': ul_bler_initial,
            'ul_bler_final': ul_bler_final,
            'dl_bler_initial': dl_bler_initial,
            'dl_bler_final': dl_bler_final,
            'ul_snr_db': ul_snr,
            'dl_snr_db': dl_snr
        }
    
    def analyze_bler_mc_averaged(self, harq_strategy: HARQStrategy,
                                app_type: str,
                                max_retx: int) -> BLERResult:
        """
        Monte Carlo averaged BLER analysis.
        
        Args:
            harq_strategy: HARQ strategy to analyze
            app_type: Application type
            max_retx: Maximum retransmissions
            
        Returns:
            BLERResult with averaged metrics
        """
        # Collect results from multiple iterations
        ul_bler_finals = []
        dl_bler_finals = []
        ul_bler_initials = []
        dl_bler_initials = []
        
        for iteration in range(self.mc_iterations):
            result = self.analyze_bler_single_iteration(harq_strategy, app_type, max_retx)
            ul_bler_finals.append(result['ul_bler_final'])
            dl_bler_finals.append(result['dl_bler_final'])
            ul_bler_initials.append(result['ul_bler_initial'])
            dl_bler_initials.append(result['dl_bler_initial'])
        
        # Calculate averages and standard deviations
        ul_bler_final_avg = np.mean(ul_bler_finals)
        dl_bler_final_avg = np.mean(dl_bler_finals)
        ul_bler_final_std = np.std(ul_bler_finals)
        dl_bler_final_std = np.std(dl_bler_finals)
        
        avg_bler = (ul_bler_final_avg + dl_bler_final_avg) / 2
        
        return BLERResult(
            harq_strategy=harq_strategy,
            app_type=app_type,
            max_retx=max_retx,
            direction="both",
            ul_bler_initial=np.mean(ul_bler_initials),
            ul_bler_final=ul_bler_final_avg,
            dl_bler_initial=np.mean(dl_bler_initials),
            dl_bler_final=dl_bler_final_avg,
            average_bler=avg_bler,
            ul_bler_std=ul_bler_final_std,
            dl_bler_std=dl_bler_final_std,
            mcs_ul=self.mcs_ul,
            mcs_dl=self.mcs_dl
        )
    
    def run_bler_analysis(self):
        """Run complete BLER analysis with Monte Carlo averaging"""
        print("\n" + "="*80)
        print("RUNNING BLER ANALYSIS (MONTE CARLO AVERAGED)")
        print("="*80)
        
        strategies = list(HARQStrategy)
        app_types = ['medium']
        
        # UL analysis (use Independent strategy as reference)
        print("\n=== UPLINK ANALYSIS ===")
        for max_retx in self.max_retransmissions_list:
            print(f"  Analyzing UL - retx={max_retx} ({self.mc_iterations} iterations)...")
            result = self.analyze_bler_mc_averaged(HARQStrategy.INDEPENDENT, 'medium', max_retx)
            result.direction = "ul"
            self.results.append(result)
        
        # DL analysis for each strategy
        print("\n=== DOWNLINK ANALYSIS ===")
        for strategy in strategies:
            if strategy == HARQStrategy.HARQ_LESS:
                # Only 1 configuration for HARQ-less (no retransmissions)
                print(f"  Analyzing {strategy.value} - no retx ({self.mc_iterations} iterations)...")
                result = self.analyze_bler_mc_averaged(strategy, 'medium', 0)
                result.direction = "dl"
                self.results.append(result)
            else:
                for max_retx in self.max_retransmissions_list:
                    print(f"  Analyzing {strategy.value} - retx={max_retx} ({self.mc_iterations} iterations)...")
                    result = self.analyze_bler_mc_averaged(strategy, 'medium', max_retx)
                    result.direction = "dl"
                    self.results.append(result)
        
        print(f"\nAnalysis completed: {len(self.results)} averaged results")
    
    def print_summary(self):
        """Print summary report"""
        print("\n" + "="*80)
        print("BLER ANALYSIS SUMMARY")
        print("="*80)
        
        app_type = 'medium'
        
        print(f"\nMedium Application Results (averaged over {self.mc_iterations} iterations):")
        print(f"{'Strategy':<15} {'Retx':<6} {'UL BLER':<20} {'DL BLER':<20}")
        print("-"*80)
        
        # Print UL results
        print("UPLINK (from Independent strategy):")
        ul_results = [r for r in self.results
                     if r.direction == "ul" and r.app_type == app_type]
        ul_results.sort(key=lambda x: x.max_retx)
        for result in ul_results:
            print(f"{'UL':<15} {result.max_retx:<6} "
                  f"{result.ul_bler_final:.2e}±{result.ul_bler_std:.1e}  {'N/A':<20}")
        
        print("\nDOWNLINK STRATEGIES:")
        for strategy in HARQStrategy:
            strategy_results = [r for r in self.results
                              if r.harq_strategy == strategy
                              and r.direction == "dl"
                              and r.app_type == app_type]
            strategy_results.sort(key=lambda x: x.max_retx)
            
            for result in strategy_results:
                retx_label = 'No Rtx' if result.max_retx == 0 else str(result.max_retx)
                print(f"{result.harq_strategy.value:<15} {retx_label:<6} "
                      f"{'N/A':<20} {result.dl_bler_final:.2e}±{result.dl_bler_std:.1e}")
            print()
        
        print("\nKEY OBSERVATIONS:")
        print("  1. Results averaged over 30 Monte Carlo iterations")
        print("  2. UL BLER decreases with retransmissions (Chase Combining)")
        print("  3. HARQ-less has no DL retransmissions")
        print("="*80)
    
    def print_harq_statistics(self):
        """Print comprehensive HARQ analysis statistics"""
        print("\n" + "="*80)
        print("HARQ ANALYSIS STATISTICS")
        print("="*80)
        
        if not self.results:
            print("No results available. Run analysis first.")
            return
        
        # Separate by direction
        ul_results = [r for r in self.results if r.direction == "ul"]
        dl_results = [r for r in self.results if r.direction == "dl"]
        
        # UL BLER
        print("\n--- UPLINK BLER (Terrestrial Anchor, Independent HARQ) ---")
        print(f"{'Retx':<8} {'Initial BLER':<15} {'Final BLER':<15} {'SNR (dB)':<12}")
        print("-"*55)
        
        ul_results_sorted = sorted(ul_results, key=lambda x: x.max_retx)
        for result in ul_results_sorted:
            print(f"{result.max_retx:<8} {result.ul_bler_initial:<15.2e} "
                  f"{result.ul_bler_final:<15.2e} {result.ul_snr_db:<12.2f}")
        
        # DL BLER by Strategy
        print("\n--- DOWNLINK BLER by HARQ Strategy ---")
        
        for strategy in [HARQStrategy.HARQ_LESS, HARQStrategy.INDEPENDENT, HARQStrategy.BUNDLED]:
            strategy_results = [r for r in dl_results if r.harq_strategy == strategy]
            strategy_results_sorted = sorted(strategy_results, key=lambda x: x.max_retx)
            
            print(f"\n{strategy.value.upper()}:")
            print(f"{'Retx':<8} {'Initial BLER':<15} {'Final BLER':<15} {'SNR (dB)':<12}")
            print("-"*55)
            
            for result in strategy_results_sorted:
                retx_label = 'No Retx' if result.max_retx == 0 else str(result.max_retx)
                print(f"{retx_label:<8} {result.dl_bler_initial:<15.2e} "
                      f"{result.dl_bler_final:<15.2e} {result.dl_snr_db:<12.2f}")
        
        # BLER Improvement Summary
        print("\n--- BLER IMPROVEMENT SUMMARY ---")
        print(f"{'Strategy':<15} {'Retx':<8} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
        print("-"*65)
        
        # UL at max retx
        ul_max = max(ul_results, key=lambda x: x.max_retx)
        ul_improvement = ul_max.ul_bler_initial / ul_max.ul_bler_final if ul_max.ul_bler_final > 0 else float('inf')
        print(f"{'UL (Indep.)':<15} {ul_max.max_retx:<8} {ul_max.ul_bler_initial:<12.2e} "
              f"{ul_max.ul_bler_final:<12.2e} {ul_improvement:<12.1f}x")
        
        # DL strategies at max retx
        for strategy in [HARQStrategy.HARQ_LESS, HARQStrategy.INDEPENDENT, HARQStrategy.BUNDLED]:
            strategy_results = [r for r in dl_results if r.harq_strategy == strategy]
            if strategy_results:
                result_max = max(strategy_results, key=lambda x: x.max_retx)
                improvement = result_max.dl_bler_initial / result_max.dl_bler_final if result_max.dl_bler_final > 0 else 1.0
                retx_label = 'No Retx' if result_max.max_retx == 0 else str(result_max.max_retx)
                print(f"{'DL ' + strategy.value:<15} {retx_label:<8} {result_max.dl_bler_initial:<12.2e} "
                      f"{result_max.dl_bler_final:<12.2e} {improvement:<12.1f}x")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    try:
        print("BLER HARQ Strategy Analyzer")
        print("="*80)
        
        analyzer = BLERHARQAnalyzer("config/parameters.yaml")
        analyzer.run_bler_analysis()
        analyzer.print_summary()
        analyzer.print_harq_statistics()
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()
