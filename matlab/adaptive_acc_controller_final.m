function [accel_cmd, headway_time, threat_level, control_mode, ...
          desired_gap_m, ttc_s, safety_state] = ...
    adaptive_acc_controller_final( ...
    ego_speed, lead_speed, gap_m, lead_valid, rf_quality, ...
    nlos_probability, packet_error_risk, snr_db, link_ok, ...
    device_confidence, aoi_s, sim_time, adaptive_enable, speed_limit)
%#codegen
% Fixed or RF-adaptive IDM-based ACC.

TS = 0.1;
MIN_STANDSTILL_GAP_M = 2.5;
HEADWAY_NOMINAL_S = 1.5;
HEADWAY_MAX_S = 2.8;
HEADWAY_FILTER_TAU_S = 0.6;
FREE_ACCEL_MPS2 = 1.4;
COMFORT_DECEL_MAG_MPS2 = 2.5;
IDM_EXPONENT = 4.0;
MAX_ACCEL_MPS2 = 1.5;
MAX_SERVICE_DECEL_MPS2 = -4.0;
MAX_EMERGENCY_DECEL_MPS2 = -7.5;
NORMAL_BRAKE_JERK_MPS3 = -3.0;
EMERGENCY_BRAKE_JERK_MPS3 = -15.0;
ACCEL_JERK_MPS3 = 2.0;

persistent headway_state accel_state previous_time

if isempty(previous_time)
    headway_state = HEADWAY_NOMINAL_S;
    accel_state = 0.0;
    previous_time = -1.0;
end

if sim_time < previous_time || ...
        (sim_time <= 1.0e-9 && previous_time > 0.0)
    headway_state = HEADWAY_NOMINAL_S;
    accel_state = 0.0;
end
previous_time = sim_time;

v = max(double(ego_speed),0.0);
vLead = max(double(lead_speed),0.0);
vSet = min(max(double(speed_limit),3.0),20.0);
hasLead = lead_valid >= 0.5;

q = min(max(double(rf_quality),0.0),1.0);
pn = min(max(double(nlos_probability),0.0),1.0);
per = min(max(double(packet_error_risk),0.0),1.0);
conf = min(max(double(device_confidence),0.0),1.0);
staleRisk = min(max(double(aoi_s),0.0),1.0);

if adaptive_enable >= 0.5
    risk = 0.45*(1.0-q) + 0.20*pn + 0.15*per + ...
           0.10*(1.0-conf) + 0.10*staleRisk;
    if link_ok < 0.5
        risk = 1.0;
    elseif conf < 0.35
        % Wrong or weak RF fingerprint: retain radar authority but command a
        % conservative time-gap policy.
        risk = max(risk,0.85);
    end
    risk = min(max(risk,0.0),1.0);
    commandedHeadway = HEADWAY_NOMINAL_S + ...
        (HEADWAY_MAX_S-HEADWAY_NOMINAL_S)*risk;
else
    risk = 0.0;
    commandedHeadway = HEADWAY_NOMINAL_S;
end

alpha = TS/(HEADWAY_FILTER_TAU_S+TS);
headway_state = headway_state + alpha*(commandedHeadway-headway_state);
headway_state = min(max(headway_state,HEADWAY_NOMINAL_S),HEADWAY_MAX_S);
headway_time = headway_state;

desired_gap_m = MIN_STANDSTILL_GAP_M+headway_time*v;

speedRatio = v/max(vSet,0.1);
aDesired = FREE_ACCEL_MPS2*(1.0-speedRatio^IDM_EXPONENT);

closingSpeed = 0.0;
ttc_s = 1.0e6;

if hasLead
    gap = max(double(gap_m),0.10);
    closingSpeed = v-vLead;

    if closingSpeed > 0.10
        ttc_s = gap/closingSpeed;
    end

    dynamicTerm = v*closingSpeed / ...
        (2.0*sqrt(FREE_ACCEL_MPS2*COMFORT_DECEL_MAG_MPS2));
    dynamicDesiredGap = MIN_STANDSTILL_GAP_M + ...
        max(0.0,v*headway_time+dynamicTerm);

    aDesired = FREE_ACCEL_MPS2 * ...
        (1.0-speedRatio^IDM_EXPONENT-(dynamicDesiredGap/gap)^2);
end

aDesired = min(max(aDesired,MAX_SERVICE_DECEL_MPS2),MAX_ACCEL_MPS2);

safety_state = 0.0;

if hasLead
    gap = max(double(gap_m),0.0);
    availableGap = max(gap-MIN_STANDSTILL_GAP_M,0.25);

    if gap <= 0.05
        safety_state = 3.0;
        aDesired = MAX_EMERGENCY_DECEL_MPS2;
    elseif ttc_s < 0.75 || gap < 0.75
        safety_state = 2.0;
        aDesired = MAX_EMERGENCY_DECEL_MPS2;
    elseif closingSpeed > 0.10
        requiredDecel = -(closingSpeed^2)/(2.0*availableGap);
        if ttc_s < 1.8 || requiredDecel < MAX_SERVICE_DECEL_MPS2
            safety_state = 2.0;
            aDesired = min(aDesired,max(requiredDecel,MAX_EMERGENCY_DECEL_MPS2));
        elseif ttc_s < 3.0
            safety_state = 1.0;
        end
    end
end

if safety_state >= 2.0
    lowerStep = accel_state+EMERGENCY_BRAKE_JERK_MPS3*TS;
else
    lowerStep = accel_state+NORMAL_BRAKE_JERK_MPS3*TS;
end
upperStep = accel_state+ACCEL_JERK_MPS3*TS;

accel_cmd = min(max(aDesired,lowerStep),upperStep);
if safety_state == 3.0
    accel_cmd = MAX_EMERGENCY_DECEL_MPS2;
end
accel_cmd = min(max(accel_cmd,MAX_EMERGENCY_DECEL_MPS2),MAX_ACCEL_MPS2);
accel_state = accel_cmd;

if safety_state >= 2.0
    control_mode = 4.0;
elseif adaptive_enable >= 0.5 && (link_ok < 0.5 || conf < 0.35)
    control_mode = 3.0;
elseif adaptive_enable >= 0.5 && hasLead && risk > 0.25
    control_mode = 2.0;
elseif hasLead
    control_mode = 1.0;
else
    control_mode = 0.0;
end

if safety_state >= 2.0
    threat_level = 4.0;
elseif adaptive_enable >= 0.5 && (link_ok < 0.5 || conf < 0.25)
    threat_level = 3.0;
elseif adaptive_enable >= 0.5 && ...
       (pn > 0.60 || q < 0.40 || snr_db < 5.0)
    threat_level = 2.0;
elseif adaptive_enable >= 0.5 && risk > 0.25
    threat_level = 1.0;
else
    threat_level = 0.0;
end
end
