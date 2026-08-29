A software-in-the-loop V2V framework integrating MATLAB/Simulink urban vehicle dynamics with a Python/PyTorch RF inference service.

The system uses scene-conditioned RF events and deep-learning inference on raw I/Q data to estimate NLOS probability and wireless-link quality. These RF metrics are then supplied to an adaptive longitudinal controller while physical gap, relative speed and TTC-based safety remain the dominant control layer.
The complete framework combines:

- MATLAB/Simulink urban vehicle and traffic simulation
- Dynamic lead-vehicle tracking
- Scene-conditioned V2V RF event activation
- ZeroMQ-based MATLAB-Python communication
- PyTorch multi-task RF inference from I/Q signals
- NLOS probability and RF-quality estimation
- RF-adaptive longitudinal vehicle control
- Clean-versus-NLOS software-in-the-loop validation

Urban Driving Scenario
        ↓
Ego Path and Lead-Vehicle Tracking
        ↓
RF Scene Context
        ↓
MATLAB → ZeroMQ → Python/PyTorch
        ↓
I/Q RF Inference
        ↓
RF Quality + NLOS Probability + SNR
        ↓
RF-Aware Longitudinal Controller
        ↓
Vehicle Speed / Gap Response

Final Simulink Architecture

The final Simulink model integrates the urban scenario, ego-path tracking, dynamic lead-vehicle selection, RF scene conditioning, live MATLAB-Python inference and adaptive longitudinal control.

RF Evaluation Zone

The RF experiment is activated using ego route position rather than an arbitrary simulation time.
RF_ZONE_START_S = 959.172 m
RF_ZONE_END_S   = 1073.437 m

Final Results
Deep-Learning RF Performance
Metric	Result
NLOS F1-score	96.69%
Balanced accuracy	96.81%
ROC-AUC	99.99%
SNR MAE	1.83 dB
Median inference latency	1.85 ms
p95 inference latency	2.26 ms
Clean vs NLOS Closed-Loop Evaluation

Metric	                       CLEAN	NLOS
MATLAB-Python link success	100%	100%
Mean RF quality	                0.7959	0.5343
Mean NLOS probability.    	0.1330	0.3093
Minimum ego speed.       	0 m/s	0 m/s
Peak deceleration      	-7.5 m/s²	-7.5 m/s²

The NLOS condition produced a clear RF-domain change while the physical vehicle controller remained bounded and stable.

Demo Videos

The v1.0 GitHub Release contains:
Full urban driving and adaptive longitudinal-braking simulation
Live Python/PyTorch RF inference activity through ZeroMQ

See:
Releases → V2V Final Demonstration
.
├── matlab/
├── python/
├── results/
├── docs/
├── LICENSE
└── README.md

Running the Physical Simulink Demo

Open MATLAB and set the Current Folder to matlab/.

Run:

prepare_ego_path

ADAPTIVE_ENABLE = 0;
RF_TRIGGER = 0;
CONDITION_CODE = 0;
ATTACK_SEVERITY = 0;
RUN_SEED = 1;
SPEED_LIMIT_MPS = 10;

open_system('v2v_urban_scenario_PHYSICAL_BASELINE')

Then run the Simulink model.

Requirements:

MATLAB
Simulink
Automated Driving Toolbox
Full RF Software-in-the-Loop System

The complete RF-aware model uses:

MATLAB/Simulink
Python
PyTorch
ZeroMQ
HDF5 RF data

MATLAB communicates with the Python inference service through:
tcp://127.0.0.1:5557

Large RF datasets and trained checkpoints are not stored directly in this repository.

Current Limitations
RF-device identification is not reliable enough for hard transmitter authentication.
Delay-spread regression is not used for safety-critical control.
RF propagation is scene-conditioned rather than full geometry-based electromagnetic ray tracing.
NLOS performance still requires validation on a fully independent external or real-world RF dataset.
Technical Report

The complete technical report is available in:

docs/V2V_RF_Adaptive_Control_FINAL.pdf

License

This project is licensed under the MIT License.
