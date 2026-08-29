function evaluate_thesis_metrics_v2(simOut)
% EVALUATE_THESIS_METRICS_V2  Production-grade V2V communications + kinematics audit.
%
%   evaluate_thesis_metrics_v2(out)
%
%   Adds on top of the v1 causality ledger:
%     - Link-health diagnostic (catches the exact "frozen at defaults" bug
%       you just hit, automatically, on every future run)
%     - Packet Delivery Ratio (PDR), Age of Information (AoI), Link Margin,
%       and NLOS/Channel-State Classification Accuracy — the four V2V KPIs
%       from Deliverable 2, computed directly from logged signals
%     - Minimum-duration event filter, so single-timestep derivative-kick
%       spikes are reported separately from genuine sustained braking
%     - V2V-to-Kinematic latency: time from RF link degradation onset to
%       physical braking onset
%     - 300 DPI multi-panel publication figure
%
% REQUIRED logsout SIGNALS (log these exact names on your canvas):
%   speed, accel, threat_level, device_confidence, nlos_prob, headway,
%   rf_quality, snr_db, link_ok, sequence
%
% ==========================================================================
clc;
if ~isprop(simOut, 'logsout')
    error('logsout missing. Enable "Log Selected Signals" on all required wires.');
end
logs = simOut.logsout;

required = {'speed','accel','threat_level','device_confidence','nlos_prob', ...
            'headway','rf_quality','snr_db','link_ok','sequence'};
available = logs.getElementNames();
missing = required(~ismember(required, available));
if ~isempty(missing)
    fprintf(2, 'Missing signals: %s\n', strjoin(missing, ', '));
    fprintf(2, 'Available: %s\n', strjoin(available, ', '));
    error('evaluate_thesis_metrics_v2:missingSignal', 'Fix signal names and re-run.');
end

t          = logs.get('speed').Values.Time;
speed      = logs.get('speed').Values.Data(:);
accel      = interp1(logs.get('accel').Values.Time,      logs.get('accel').Values.Data(:),      t, 'linear', 'extrap');
threat     = interp1(logs.get('threat_level').Values.Time, logs.get('threat_level').Values.Data(:), t, 'previous', 'extrap');
dev_conf   = interp1(logs.get('device_confidence').Values.Time, logs.get('device_confidence').Values.Data(:), t, 'linear', 'extrap');
nlos       = interp1(logs.get('nlos_prob').Values.Time,   logs.get('nlos_prob').Values.Data(:),   t, 'linear', 'extrap');
headway    = interp1(logs.get('headway').Values.Time,     logs.get('headway').Values.Data(:),     t, 'linear', 'extrap');
rf_qual    = interp1(logs.get('rf_quality').Values.Time,  logs.get('rf_quality').Values.Data(:),  t, 'linear', 'extrap');
snr_db     = interp1(logs.get('snr_db').Values.Time,      logs.get('snr_db').Values.Data(:),      t, 'linear', 'extrap');
link_ok    = interp1(logs.get('link_ok').Values.Time,     logs.get('link_ok').Values.Data(:),     t, 'previous', 'extrap');
sequence   = interp1(logs.get('sequence').Values.Time,    logs.get('sequence').Values.Data(:),    t, 'previous', 'extrap');
Ts = median(diff(t));

% ==========================================================================
% 0. LINK-HEALTH SANITY CHECK — run this before trusting anything else
% ==========================================================================
fprintf('========================================================================================\n');
fprintf('                              LINK HEALTH DIAGNOSTIC\n');
fprintf('========================================================================================\n');
link_up_pct = 100 * mean(link_ok);
fprintf('ZMQ link successful-response rate: %.1f%% of calls\n', link_up_pct);
if link_up_pct < 50
    fprintf(2, ['*** WARNING: link_ok < 50%%. rf_quality/nlos/snr are mostly fail-safe\n' ...
                '    defaults, NOT real inference results. Increase ZMQ_RCVTIMEO further,\n' ...
                '    confirm infer_service.py is actually running and bound to port 5557,\n' ...
                '    and check the VS Code terminal is printing incoming requests. ***\n']);
end
fprintf('----------------------------------------------------------------------------------------\n\n');

% ==========================================================================
% 1. EVENT-BASED CAUSALITY LEDGER (with spike filtering)
% ==========================================================================
MIN_EVENT_SAMPLES = 3;   % require >=3 consecutive samples to count as a real event
                          % (filters single-timestep derivative-kick noise —
                          % see find_lead_vehicle target-switching note below)

is_braking = (accel < -0.5);
starts = find(diff([0; is_braking]) == 1);
ends   = find(diff([is_braking; 0]) == -1);
durations_samples = ends - starts + 1;

valid = durations_samples >= MIN_EVENT_SAMPLES;
spike_count = sum(~valid);
starts = starts(valid);
ends   = ends(valid);
num_events = numel(starts);

cyber_events = 0; traffic_events = 0;
fprintf('========================================================================================\n');
fprintf('                    RETARDATION CAUSALITY LEDGER (spike-filtered, min %d samples)\n', MIN_EVENT_SAMPLES);
fprintf('========================================================================================\n');
fprintf('Evt # |  Time Window (s)  | Peak Decel | Min Headway | Min RF Qual | Max NLOS | Primary Root Cause\n');
fprintf('----------------------------------------------------------------------------------------\n');
for i = 1:num_events
    idx = starts(i):ends(i);
    t_start = t(starts(i)); t_end = t(ends(i));
    peak_decel = min(accel(idx));
    min_hw = min(headway(idx));
    min_rf = min(rf_qual(idx));
    max_nlos = max(nlos(idx));
    max_thr = max(threat(idx));

    if min_rf < 0.6 || max_nlos > 0.5 || max_thr > 0
        cause = 'CYBER THREAT (RF Drop / NLOS AI Trigger)';
        cyber_events = cyber_events + 1;
    elseif min_hw < 12.0
        cause = 'KINEMATIC GAP (Traffic Radar Follow)';
        traffic_events = traffic_events + 1;
    else
        cause = 'SPEED REGULATION (Trajectory Curve / Cruise)';
    end
    fprintf('%5d | %6.1f - %-6.1f s | %8.2f   | %9.2f s | %10.2f  | %7.2f  | %s\n', ...
        i, t_start, t_end, peak_decel, min_hw, min_rf, max_nlos, cause);
end
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('Filtered out %d single/double-sample spikes (< %d samples) — see Sec. 5 note.\n\n', ...
        spike_count, MIN_EVENT_SAMPLES);

% ==========================================================================
% 2. V2V COMMUNICATION KPIs (Deliverable 2)
% ==========================================================================
fprintf('========================================================================================\n');
fprintf('                         V2V COMMUNICATION-LAYER KPIs\n');
fprintf('========================================================================================\n');

% --- Packet Delivery Ratio (PDR) ---
% PDR = successfully-acknowledged requests / total requests attempted.
% link_ok double as an indicator of successful round-trip per call.
pdr_pct = 100 * mean(link_ok);
fprintf('Packet Delivery Ratio (PDR):        %.2f %%\n', pdr_pct);

% --- Age of Information (AoI) ---
% AoI(t) = elapsed time since the last successfully-received packet.
% Grows linearly while link_ok=0, resets to ~Ts on each success.
aoi = zeros(size(t));
last_success_t = t(1);
for k = 1:numel(t)
    if link_ok(k) >= 0.5
        last_success_t = t(k);
    end
    aoi(k) = t(k) - last_success_t;
end
fprintf('Mean Age of Information (AoI):      %.3f s\n', mean(aoi));
fprintf('Peak Age of Information (AoI):      %.3f s\n', max(aoi));

% --- Link Margin ---
% Margin above the minimum SNR the controller trusts (5 dB per your
% adaptive_braking_controller.m comfort-clamp threshold).
SNR_MIN_DB = 5.0;
link_margin_db = snr_db - SNR_MIN_DB;
fprintf('Mean Link Margin (SNR - %.0f dB):     %.2f dB\n', SNR_MIN_DB, mean(link_margin_db));
fprintf('Minimum Link Margin observed:        %.2f dB\n', min(link_margin_db));
fprintf('Time with Negative Margin (outage): %.2f s (%.1f%% of run)\n', ...
        sum(link_margin_db < 0) * Ts, 100 * mean(link_margin_db < 0));

% --- NLOS / Channel-State Classification Accuracy ---
% Requires a ground-truth NLOS injection window — set these to your
% scenario's actual blockage timestamps.
NLOS_GT_START = 15.0;
NLOS_GT_END   = 25.0;
gt_nlos = (t >= NLOS_GT_START) & (t <= NLOS_GT_END);
pred_nlos = nlos > 0.5;
channel_accuracy_pct = 100 * mean(pred_nlos == gt_nlos);
fprintf('NLOS/Channel-State Classification Accuracy: %.2f %% (GT window %.0f-%.0fs)\n', ...
        channel_accuracy_pct, NLOS_GT_START, NLOS_GT_END);
fprintf('----------------------------------------------------------------------------------------\n\n');

% ==========================================================================
% 3. V2V-TO-KINEMATIC LATENCY
% ==========================================================================
DEGRADATION_RF_THRESH = 0.6;
degrade_idx = find(rf_qual < DEGRADATION_RF_THRESH, 1, 'first');
if ~isempty(degrade_idx) && num_events > 0
    t_degrade = t(degrade_idx);
    next_event = find(t(starts) >= t_degrade, 1, 'first');
    if ~isempty(next_event)
        v2v_latency_ms = (t(starts(next_event)) - t_degrade) * 1000;
        fprintf('V2V-to-Kinematic Latency (RF drop -> brake onset): %.1f ms\n\n', v2v_latency_ms);
    else
        fprintf('RF degradation detected but no subsequent braking event found.\n\n');
    end
else
    fprintf('No RF degradation below threshold %.2f detected in this run.\n\n', DEGRADATION_RF_THRESH);
end

% ==========================================================================
% 4. SUMMARY KPIs
% ==========================================================================
total_time = t(end) - t(1);
braking_time = sum(is_braking) * Ts;
fprintf('========================================================================================\n');
fprintf('                         CYBER-PHYSICAL SUMMARY\n');
fprintf('========================================================================================\n');
fprintf('Total Simulation Time:        %.2f s\n', total_time);
fprintf('Valid Retardation Events:     %d (filtered %d spikes)\n', num_events, spike_count);
fprintf('Time Spent Braking:           %.2f s (%.1f%% of mission)\n', braking_time, 100*braking_time/total_time);
fprintf('Cyber/RF-Triggered Braking:   %d events (%.1f%%)\n', cyber_events, 100*cyber_events/max(num_events,1));
fprintf('Traffic-Triggered Braking:    %d events (%.1f%%)\n', traffic_events, 100*traffic_events/max(num_events,1));
fprintf('Absolute Peak Retardation:    %.2f m/s^2\n', min(accel));
fprintf('========================================================================================\n\n');

% ==========================================================================
% 5. 300 DPI PUBLICATION FIGURE
% ==========================================================================
fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 12 9]);
tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, 'V2V Communication-Governed Adaptive Braking: Full System Audit', ...
      'FontWeight', 'bold', 'FontSize', 13);

nexttile(tl,1); plot(t, speed, 'b-', 'LineWidth', 1.4);
xlabel('Time (s)'); ylabel('Speed (m/s)'); title('Velocity'); grid on;

nexttile(tl,2); plot(t, accel, 'r-', 'LineWidth', 1.4); hold on;
yline(-0.5, 'k:', 'Event threshold');
xlabel('Time (s)'); ylabel('accel (m/s^2)'); title('Acceleration'); grid on;

nexttile(tl,3); plot(t, snr_db, 'g-', 'LineWidth', 1.4); hold on;
yline(SNR_MIN_DB, 'r--', 'Min trusted SNR');
xlabel('Time (s)'); ylabel('SNR (dB)'); title('Link SNR'); grid on;

nexttile(tl,4); plot(t, aoi, 'm-', 'LineWidth', 1.4);
xlabel('Time (s)'); ylabel('AoI (s)'); title('Age of Information'); grid on;

nexttile(tl,5); plot(t, link_ok, 'k-', 'LineWidth', 1.2);
ylim([-0.1 1.1]); xlabel('Time (s)'); ylabel('Link OK (0/1)'); title('Link Health'); grid on;

nexttile(tl,6); plot(t, nlos, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.4); hold on;
plot(t, pred_nlos, 'b--', 'LineWidth', 1.0);
legend('P(NLOS)', 'Classified NLOS', 'Location', 'best');
xlabel('Time (s)'); title('NLOS Estimate vs Classification'); grid on;

exportgraphics(fig, 'results/thesis_v2v_full_audit.png', 'Resolution', 300);
fprintf('Saved: results/thesis_v2v_full_audit.png (300 DPI)\n');

end