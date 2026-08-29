function [attack_active, attack_code, attack_gt, ...
          expected_device_id, scene_nlos, tx_valid] = ...
    rf_scene_context_dynamic(path_s, lead_actor_id, lead_valid, gap_m, ...
                             condition_code, severity, ...
                             zone_start_s, zone_end_s)
%#codegen
%RF_SCENE_CONTEXT_DYNAMIC
% Scene-conditioned RF experiment logic for the dense second roundabout.
%
% The CURRENT valid lead vehicle is treated as the V2V transmitter.
%
% Inputs:
%   path_s            Ego route position (m)
%   lead_actor_id     Current lead ActorID
%   lead_valid        Current tracker-valid flag
%   gap_m             Current ego-to-lead gap (m)
%   condition_code    0 clean, 1 NLOS, 2 jamming, 3 spoofing
%   severity          Experiment severity in [0,1]
%   zone_start_s      Calibrated RF-zone start path position (m)
%   zone_end_s        Calibrated RF-zone end path position (m)
%
% Outputs:
%   attack_active      RF impairment active flag
%   attack_code        0 outside zone; otherwise selected condition
%   attack_gt          Ground-truth impairment flag for logging only
%   expected_device_id Expected RF device class for current lead
%   scene_nlos         Ground-truth urban-zone flag for logging only
%   tx_valid           A valid V2V transmitting lead is present

MAX_RF_GAP_M = 60.0;
NUM_RF_DEVICE_CLASSES = 3.0;

s0 = min(double(zone_start_s), double(zone_end_s));
s1 = max(double(zone_start_s), double(zone_end_s));

actor_id = floor(double(lead_actor_id));
gap = double(gap_m);

tx_valid_bool = ...
    (double(lead_valid) >= 0.5) && ...
    (actor_id >= 1.0) && ...
    isfinite(gap) && ...
    (gap > 0.0) && ...
    (gap < MAX_RF_GAP_M);

inside_zone = ...
    (double(path_s) >= s0) && ...
    (double(path_s) <= s1);

scene_nlos_bool = tx_valid_bool && inside_zone;

if tx_valid_bool
    expected_device_id = mod(actor_id - 1.0, NUM_RF_DEVICE_CLASSES);
else
    expected_device_id = 0.0;
end

code = floor(double(condition_code));
code = min(max(code,0.0),3.0);

active = ...
    (code > 0.0) && ...
    (double(severity) > 0.0) && ...
    scene_nlos_bool;

tx_valid = double(tx_valid_bool);
scene_nlos = double(scene_nlos_bool);

if active
    attack_active = 1.0;
    attack_code = code;
    attack_gt = 1.0;
else
    attack_active = 0.0;
    attack_code = 0.0;
    attack_gt = 0.0;
end
end
