# RF-Aware V2V Communication and Adaptive Vehicle Control

A software-in-the-loop V2V framework integrating MATLAB/Simulink vehicle dynamics with a Python/PyTorch RF inference service for scene-conditioned V2V communication and adaptive longitudinal control.

The system estimates wireless-link quality and Non-Line-of-Sight (NLOS) probability from raw I/Q signals and supplies these RF metrics to the vehicle controller. Physical gap, relative speed and TTC-based safety remain the primary control layer.

---

## System Overview

The framework integrates:

- Urban driving simulation in MATLAB/Simulink
- Dynamic lead-vehicle tracking
- Scene-conditioned V2V RF events
- MATLAB-Python communication through ZeroMQ
- PyTorch inference on raw I/Q signals
- NLOS and RF-quality estimation
- RF-aware longitudinal control
- CLEAN-versus-NLOS closed-loop evaluation

### End-to-End Flow

```text
Urban Driving Scenario
        ↓
Ego Path + Lead-Vehicle Tracking
        ↓
RF Scene Context
        ↓
MATLAB / Simulink
        ↓
ZeroMQ
        ↓
Python / PyTorch RF Inference
        ↓
RF Quality + NLOS Probability + SNR
        ↓
RF-Aware Longitudinal Controller
        ↓
Vehicle Speed and Gap Response
```

## Final Simulink Architecture

The final Simulink model combines the physical driving scenario, ego-path tracking, dynamic lead-vehicle selection, RF scene conditioning, live RF inference and adaptive longitudinal control.

## Urban Driving Scenario

The ego vehicle follows a predefined urban route containing intersections, highway sections, roundabouts and multiple traffic actors.

A building-heavy section of the route is used as the RF evaluation region.

The RF experiment is activated using ego route position rather than an arbitrary simulation time.

RF_ZONE_START_S = 959.172 m
RF_ZONE_END_S   = 1073.437 m
## RF Intelligence Pipeline

The Python/PyTorch inference service processes raw I/Q frames and returns RF estimates to Simulink through ZeroMQ.

Important outputs include:

NLOS probability
RF quality
Estimated SNR
Packet-error risk
Device confidence
Delay-spread estimate
Inference latency
SIL sequence and link status

RF information acts as a supervisory input. Physical gap and TTC-based safety remain active independently of the RF subsystem.

## Deep-Learning Performance
Metric	Result
NLOS F1-score	96.69%
NLOS balanced accuracy	96.81%
NLOS ROC-AUC	99.99%
SNR MAE	1.83 dB
Median inference latency	1.85 ms
p95 inference latency	2.26 ms

The reported values correspond to evaluation on the finite RF dataset used in the project.

## CLEAN vs NLOS Closed-Loop Evaluation

CLEAN and NLOS experiments were compared using ego route position (path_s) so that both runs are evaluated at the same physical locations.

Metric	CLEAN	NLOS
MATLAB-Python link success	100%	100%
Mean RF quality in RF zone	0.7959	0.5343
Mean NLOS probability in RF zone	0.1330	0.3093
Minimum ego speed	0 m/s	0 m/s
Peak deceleration	-7.5 m/s²	-7.5 m/s²

The NLOS condition reduced mean RF quality by approximately 0.2615 and increased mean NLOS probability by approximately 0.1763.

The communication-layer change is clearly detected while the physical safety controller keeps the longitudinal response bounded.

## RF-Only Validation

Before enabling RF-aware vehicle adaptation, the RF pipeline was validated independently.

MATLAB-Python SIL link success    = 100%

Mean RF quality outside RF zone  = 0.7917
Mean RF quality inside RF zone   = 0.1155

Mean NLOS outside RF zone        = 0.1339
Mean NLOS inside RF zone         = 0.5941

This confirms that the scene-conditioned RF region produces the intended communication degradation before RF information influences vehicle control.

## Demo Video

The v1.0 – V2V Final Demonstration GitHub Release contains the integrated final simulation video.

The video shows:

the urban driving scenario running in MATLAB/Simulink,
ego-vehicle longitudinal control and adaptive braking,
the RF evaluation region,
live MATLAB-Python communication through ZeroMQ,
and the Python/PyTorch inference server processing RF requests during the simulation.

This provides an end-to-end demonstration of the vehicle-control and RF-inference pipeline operating together.



## Repository Structure

```text
RF-Aware-V2V-Communication-and-Adaptive-Vehicle-Control/
├── matlab/
│   ├── Simulink vehicle models
│   ├── Urban driving scenario
│   ├── Ego-path generation
│   ├── Lead-vehicle tracking
│   ├── RF scene conditioning
│   ├── ZeroMQ interface
│   └── Evaluation scripts
│
├── python/
│   ├── RF dataset generation
│   ├── Preprocessing
│   ├── PyTorch training
│   ├── RF inference service
│   └── Model evaluation
│
├── results/
│   └── Final simulation and evaluation figures
│
├── docs/
│   └── Final technical report
│
├── LICENSE
└── README.md
```

---

## Running the Physical Simulink Demo

### Requirements

- MATLAB
- Simulink
- Automated Driving Toolbox

Open MATLAB and set the **Current Folder** to:

```text
matlab/
```

Then run:

```matlab
prepare_ego_path

ADAPTIVE_ENABLE = 0;
RF_TRIGGER = 0;
CONDITION_CODE = 0;
ATTACK_SEVERITY = 0;
RUN_SEED = 1;
SPEED_LIMIT_MPS = 10;

open_system('v2v_urban_scenario_PHYSICAL_BASELINE')
```

Then press **Run** in Simulink.
## Full RF Software-in-the-Loop System

The complete RF-aware configuration additionally requires:

Python
PyTorch
ZeroMQ
HDF5 RF data

MATLAB communicates with the Python inference server through:

tcp://127.0.0.1:5557

The final RF model is:

matlab/v2v_urban_scenario_RF_FINAL.slx

Large RF datasets and trained checkpoints are intentionally not stored directly in this repository.

## Current Limitations
RF-device identification is not sufficiently reliable for hard transmitter authentication.
Delay-spread regression is not used for safety-critical vehicle control.
RF propagation is scene-conditioned using RF data rather than full geometry-based electromagnetic ray tracing.
NLOS performance requires further validation on a completely independent or real-world RF dataset.
## Technical Report

The complete project report is available at:

docs/V2V_RF_Adaptive_Control_FINAL.pdf

## Technologies

MATLAB · Simulink · Automated Driving Toolbox · Python · PyTorch · ZeroMQ · HDF5 · Deep Learning · V2V · Software-in-the-Loop
## License

This project is released under the MIT License.
