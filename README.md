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
