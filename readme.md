# TN-NTN Multi-Connectivity Simulator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulator for energy-efficient uplink-downlink decoupling in 6G TN-NTN networks.

## Capabilities

**Power Analysis:**
- Required UL transmit power across terrestrial/UAV/HAPS/satellite
- Energy per bit and per transport block
- Power gap quantification (10+ dB TN vs satellite)
- Battery lifetime projections

**HARQ Analysis:**
- Three strategies: HARQ-less, Independent, Bundled
- BLER performance with Chase Combining
- Power reduction evaluation (up to 40%)
- Bundling delay overhead (<10% RTT)

**System Features:**
- Monte Carlo statistical analysis (100+ iterations)
- 3GPP-compliant channel models (TR 38.901)
- LEO constellation simulation (Starlink/Kuiper)
- 5G NR transport block calculations

## Requirements

Python 3.8+, NumPy, Pandas, Matplotlib, PyYAML, SciPy, Seaborn

## Quick Start

```bash
# 1. Basic power analysis 
python basic_power_analyzer.py

# 2. HARQ BLER analysis 
python bler_harq_analyzer.py

# 3. Activity factor analysis
python dc_harq_analyzer.py

# 4. Bundling delay analysis 
python bundle_delay_analyzer.py
```

## Configuration

Edit `config/parameters.yaml`:

```yaml
# =============================================================================
# UE (USER EQUIPMENT) CONFIGURATION
# =============================================================================
ue:
  location: [48.8566, 2.3522, 0]   # [lat, lon, alt] - Paris, France
  max_tx_power_dbm: 23.0           # Maximum UE transmit power (3GPP Class 3)
  antenna_gain_dbi: 0.0            # UE antenna gain
  circuit_power_mw: 150.0          # Circuit power consumption
  rx_power_mw: 50.0                # Reception power consumption

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
constants:
  speed_of_light: 299792458        # Speed of light in m/s
  earth_radius_km: 6371            # Earth radius in km

# =============================================================================
# CHANNEL CONFIGURATION
# =============================================================================
channel:
  noise_figure_db: 7.0             # Receiver noise figure
  thermal_noise_density_dbm_hz: -174.0  # Thermal noise density
  target_snr_db:
    ul_data: 10.0                  # Uplink data SNR target
    dl_data: 15.0                  # Downlink data SNR target

# =============================================================================
# TERRESTRIAL NETWORK CONFIGURATION
# 3GPP Band n78 (3.5 GHz) - Primary 5G NR band
# =============================================================================
terrestrial:
  frequency_ghz: 3.5               # Operating frequency
  bandwidth_mhz: 20                # Channel bandwidth
  tx_power_dbm: 30.0               # Base station transmit power
  antenna_gain_dbi: 18.0           # Base station antenna gain
  shadowing_std_db: 8.0            # Log-normal shadowing standard deviation
  processing_delay_ms: 2.0         # Base station processing delay
  base_stations:
    - [48.8656, 2.3522, 30]        # [lat, lon, alt] - Base station location

# =============================================================================
# UAV CONFIGURATION
# 3GPP Band 1 (2.1 GHz) - Air-to-ground communications
# =============================================================================
...
```

## Usage

### 1. Basic Power Analysis

Calculates required UL transmit power and energy per bit for all access types.

```bash
python basic_power_analyzer.py
```

**What you get:**
```
MONTE CARLO STATISTICS (100 iterations)
Access       Required Power  Feasible  Energy/bit   Lifetime vs Sat
terrestrial  15.2 dBm       Yes       0.082 μJ     11.8×
satellite    25.7 dBm       No        0.967 μJ     1.0×
```

**Key result:** 10.5 dB power gap (terrestrial vs satellite)

### 2. BLER HARQ Analysis

Compares HARQ strategies across retransmissions 1-8.

```bash
python bler_harq_analyzer.py
```

**What it analyzes:**
- UL BLER with Chase Combining (8 retransmission levels)
- DL HARQ-less (no retransmissions)
- DL Independent HARQ (per-TB feedback)
- DL Bundled HARQ (aggregated feedback)

**Key result:** Chase Combining reduces BLER by 10-100× depending on retransmissions

### 3. Activity Factor Analysis

Sweeps UE activity 0-100% for two traffic ratios.

```bash
python dc_harq_analyzer.py
```

**What it does:**
- Analyzes power vs activity factor (0%, 5%, ..., 100%)
- Two traffic scenarios: 10UL-90DL and 50UL-50DL
- Compares HARQ-less, Independent, and Bundled (sizes 2-14)

**Outputs statistics for 60% activity:**
```
STATISTICS FOR 10UL-90DL (at 60% activity)
Strategy             UL Data      DL Data      UL HARQ     DL HARQ     Total
harq_less            45.2 mW      78.5 mW      2.1 mW      0.0 mW      125.8 mW
independent          45.2 mW      78.5 mW      2.1 mW      35.4 mW     161.2 mW
bundled_2            45.2 mW      78.5 mW      2.1 mW      17.7 mW     143.5 mW
```

**Key result:** Bundled HARQ saves up to 40% power vs Independent

### 4. Bundling Delay Analysis

Calculates bundling delay normalized to satellite RTT.

```bash
python bundle_delay_analyzer.py
```

**What it does:**
- Samples 100 RTT values from different satellite elevations
- Calculates bundling delay for sizes 2-14
- Normalizes delay as % of RTT

**Sample output:**
```
BUNDLE DELAY ANALYSIS SUMMARY
Bundle   Abs Delay    Median %RTT  P25 %RTT    P75 %RTT    P95 %RTT
2        0.25 ms      2.3%         1.8%        2.9%        3.5%
4        0.75 ms      5.8%         4.5%        7.2%        8.6%
8        1.75 ms      9.2%         7.1%        11.5%       13.8%
14       3.25 ms      12.1%        9.4%        15.1%       18.2%
```

**Key result:** Bundling delay <10% RTT for bundle sizes ≤12

## Code Structure

```
simulator/
├── config/parameters.yaml        # Configuration (frequencies, positions, etc.)
├── basic_power_analyzer.py       # Power/energy analysis with MC
├── bler_harq_analyzer.py         # HARQ BLER performance
├── dc_harq_analyzer.py           # Activity factor sweep
├── bundle_delay_analyzer.py      # Bundling delay overhead
├── power_models.py               # TB-level power calculations
├── channel_models.py             # 3GPP path loss (TR 38.901)
├── constellation.py              # LEO orbital mechanics
└── geometry_utils.py             # Haversine + 3D distance
```

**Core modules:**
- `power_models.py`: Link budget → transmit power (SNR-target or power-constrained)
- `channel_models.py`: Path loss for terrestrial/UAV/HAPS/satellite
- `constellation.py`: Starlink/Kuiper orbit simulation with handovers
- `geometry_utils.py`: Lat/lon/alt → 3D distance + elevation/azimuth

## Technical Details

**Channel models (3GPP-compliant):**
- Terrestrial: TR 38.901 UMa/UMi/RMa + log-normal shadowing
- UAV: Probabilistic LoS (elevation-dependent)
- HAPS: NTN A2G model (stratospheric propagation)
- Satellite: Free-space + 0.5 dB atmospheric loss

**5G NR physical layer:**
- Numerology μ=1 → slot duration 0.5 ms (30 kHz SCS)
- MCS table: QPSK/16QAM/64QAM (3GPP TS 38.214)
- Transport Block Sizing: MCS-based (not hardcoded)
- HARQ Chase Combining: SNR gain = 10·log₁₀(num_transmissions)

**LEO constellation:**
- Starlink: 72 planes × 22 sats @ 550 km, 53° inclination
- Orbital period: ~95 minutes
- Handover: 20° elevation threshold
- Auto-update: Background thread for satellite motion


## Results

**Power gap (terrestrial vs satellite):**
```
Terrestrial UL: 15.2 dBm
Satellite UL:   25.7 dBm
Gap:            10.5 dB (distance-dominated)
```

**Energy savings:**
```
Access       Energy/bit   Lifetime vs Satellite
Terrestrial  0.082 μJ     11.8×
UAV          0.065 μJ     14.9×
HAPS         0.142 μJ     6.8×
Satellite    0.967 μJ     1.0× (baseline)
```

**HARQ bundling (bundle size=8):**
```
Power reduction: 35% vs Independent
Delay overhead:  5.8% of RTT (median)
```

## License

MIT License
