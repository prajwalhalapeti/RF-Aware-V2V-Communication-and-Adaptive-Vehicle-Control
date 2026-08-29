function [rf_quality, device_confidence, snr_db, packet_error_risk, ...
          delay_spread, nlos_probability, link_ok, wireless_packet_ok, ...
          sequence, transmitter_id, gt_device_id, inference_ms, gt_nlos, ...
          nlos_raw, rf_quality_raw, actual_hdf5_idx, attack_gt, aoi_s] = ...
    rf_metrics_receiver_experiment( ...
          packet_index, sim_time, attack_code, attack_active, severity, ...
          run_seed, expected_device_id, trigger)
%#codegen
%RF_METRICS_RECEIVER_EXPERIMENT Separate SIL health from wireless delivery.
%
% link_ok:
%   Valid localhost MATLAB/Python transaction.
%
% wireless_packet_ok:
%   Simulated V2V delivery result generated from the channel/PER model.
%
% The raw CNN outputs are always logged on a valid SIL response. Controller
% metrics update only when the simulated wireless packet is delivered.

coder.extrinsic('v2v_zmq_request_experiment');
coder.extrinsic('decode_v2v_json_experiment');

persistent haveDelivered lastQ lastConf lastSNR lastPER lastDS lastNLOS
persistent lastServerSequence lastDeliveredTime previousPacketIndex

if isempty(previousPacketIndex)
    haveDelivered = false;
    lastQ = 0.25;
    lastConf = 0.0;
    lastSNR = -5.0;
    lastPER = 1.0;
    lastDS = 1.0e-6;
    lastNLOS = 0.5;
    lastServerSequence = -1.0;
    lastDeliveredTime = double(sim_time);
    previousPacketIndex = -1.0;
end

if double(packet_index) < previousPacketIndex
    haveDelivered = false;
    lastServerSequence = -1.0;
    lastDeliveredTime = double(sim_time);
end
previousPacketIndex = double(packet_index);

% Conservative finite defaults.
rf_quality = 0.25;
device_confidence = 0.0;
snr_db = -5.0;
packet_error_risk = 1.0;
delay_spread = 1.0e-6;
nlos_probability = 0.5;
link_ok = 0.0;
wireless_packet_ok = 0.0;
sequence = lastServerSequence;
transmitter_id = -1.0;
gt_device_id = -1.0;
inference_ms = 0.0;
gt_nlos = 0.0;
nlos_raw = 0.5;
rf_quality_raw = 0.25;
actual_hdf5_idx = -1.0;
attack_gt = double(attack_active >= 0.5);
aoi_s = max(double(sim_time)-lastDeliveredTime,0.0);

if trigger < 0.5
    return;
end

raw = '';
raw = v2v_zmq_request_experiment(packet_index,sim_time, ...
    attack_code,attack_active,severity,run_seed,expected_device_id);

if isempty(raw)
    if haveDelivered
        rf_quality = min(lastQ,0.35);
        device_confidence = min(lastConf,0.25);
        snr_db = lastSNR;
        packet_error_risk = max(lastPER,0.80);
        delay_spread = lastDS;
        nlos_probability = lastNLOS;
    end
    return;
end

values = zeros(1,19);
values = decode_v2v_json_experiment(raw);

if values(1) < 0.5
    return;
end

newSequence = values(8);
if lastServerSequence >= 0.0 && newSequence <= lastServerSequence
    return;
end

link_ok = 1.0;
lastServerSequence = newSequence;
sequence = newSequence;

rf_quality_raw = values(2);
nlos_raw = values(7);
transmitter_id = values(9);
gt_device_id = values(10);
inference_ms = values(11);
gt_nlos = values(12);
wireless_packet_ok = values(13);
actual_hdf5_idx = values(14);
attack_gt = values(16);

if wireless_packet_ok >= 0.5
    rf_quality = values(2);

    % Convert raw classifier confidence into identity trust. A highly
    % confident prediction for the wrong transmitter must not be treated as
    % trusted V2V identity.
    predicted_id = values(9);
    expected_id = values(18);
    if round(predicted_id) == round(expected_id)
        device_confidence = values(3);
    else
        device_confidence = 0.0;
    end

    snr_db = values(4);
    packet_error_risk = values(5);
    delay_spread = values(6);
    nlos_probability = values(7);

    lastQ = rf_quality;
    lastConf = device_confidence;
    lastSNR = snr_db;
    lastPER = packet_error_risk;
    lastDS = delay_spread;
    lastNLOS = nlos_probability;
    lastDeliveredTime = double(sim_time);
    haveDelivered = true;
elseif haveDelivered
    rf_quality = lastQ;
    device_confidence = lastConf;
    snr_db = lastSNR;
    packet_error_risk = lastPER;
    delay_spread = lastDS;
    nlos_probability = lastNLOS;
end

aoi_s = max(double(sim_time)-lastDeliveredTime,0.0);
end
