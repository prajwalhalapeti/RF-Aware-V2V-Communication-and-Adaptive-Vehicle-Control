function [lead_speed, gap_m, lead_actor_id, lead_valid, range_rate] = ...
    find_lead_vehicle_final(scenario_data, ego_speed)
%#codegen
% Generic nearest in-lane lead tracker for Scenario Reader VEHICLE coordinates.

TS = 0.1;
MAX_LOOK_AHEAD_M = 100.0;
MIN_FORWARD_ORIGIN_M = 1.0;
BASE_LATERAL_GATE_M = 1.9;
LATERAL_GATE_SLOPE = 0.020;
MAX_LATERAL_GATE_M = 3.5;
MAX_NEW_TARGET_YAW_DEG = 50.0;
MAX_LOCKED_TARGET_YAW_DEG = 70.0;
PERSISTENCE_SAMPLES = 3.0;
LOSS_GRACE_SAMPLES = 3.0;
SWITCH_ADVANTAGE_M = 1.0;
ORIGIN_TO_BUMPER_M = 4.2;
RANGE_RATE_ALPHA = 0.35;
MAX_ABS_RANGE_RATE_MPS = 40.0;

persistent locked_id locked_lon locked_speed
persistent pending_id pending_count lost_count
persistent previous_lon filtered_range_rate had_range previous_time

sim_time = double(scenario_data.Time);

if isempty(previous_time)
    locked_id = -1.0;
    locked_lon = MAX_LOOK_AHEAD_M + ORIGIN_TO_BUMPER_M;
    locked_speed = max(double(ego_speed),0.0);
    pending_id = -1.0;
    pending_count = 0.0;
    lost_count = 0.0;
    previous_lon = 0.0;
    filtered_range_rate = 0.0;
    had_range = false;
    previous_time = -1.0;
end

if sim_time < previous_time || ...
        (sim_time <= 1.0e-9 && previous_time > 0.0)
    locked_id = -1.0;
    locked_lon = MAX_LOOK_AHEAD_M + ORIGIN_TO_BUMPER_M;
    locked_speed = max(double(ego_speed),0.0);
    pending_id = -1.0;
    pending_count = 0.0;
    lost_count = 0.0;
    previous_lon = 0.0;
    filtered_range_rate = 0.0;
    had_range = false;
end
previous_time = sim_time;

actors = scenario_data.Actors;
n = min(int32(double(scenario_data.NumActors)),int32(numel(actors)));

raw_id = -1.0;
raw_lon = MAX_LOOK_AHEAD_M + ORIGIN_TO_BUMPER_M;
raw_speed = max(double(ego_speed),0.0);

locked_visible = false;
locked_lon_now = locked_lon;
locked_speed_now = locked_speed;

for i = 1:n
    actor_id = double(actors(i).ActorID);
    if actor_id <= 0.0
        continue;
    end

    lon = double(actors(i).Position(1));
    lat = double(actors(i).Position(2));
    relative_yaw = mod(double(actors(i).Yaw)+180.0,360.0)-180.0;

    lateral_gate = min(MAX_LATERAL_GATE_M, ...
        BASE_LATERAL_GATE_M + LATERAL_GATE_SLOPE*max(lon,0.0));

    actor_speed = hypot(double(actors(i).Velocity(1)), ...
                        double(actors(i).Velocity(2)));

    new_valid = lon > MIN_FORWARD_ORIGIN_M && ...
        lon < MAX_LOOK_AHEAD_M + ORIGIN_TO_BUMPER_M && ...
        abs(lat) < lateral_gate && ...
        abs(relative_yaw) < MAX_NEW_TARGET_YAW_DEG;

    if actor_id == locked_id
        locked_valid = lon > 0.0 && ...
            lon < MAX_LOOK_AHEAD_M + ORIGIN_TO_BUMPER_M + 6.0 && ...
            abs(lat) < lateral_gate + 0.8 && ...
            abs(relative_yaw) < MAX_LOCKED_TARGET_YAW_DEG;

        if locked_valid
            locked_visible = true;
            locked_lon_now = lon;
            locked_speed_now = actor_speed;
        end
    end

    if new_valid && lon < raw_lon
        raw_id = actor_id;
        raw_lon = lon;
        raw_speed = actor_speed;
    end
end

if locked_id < 0.0
    if raw_id >= 0.0
        locked_id = raw_id;
        locked_lon = raw_lon;
        locked_speed = raw_speed;
    end
    pending_id = -1.0;
    pending_count = 0.0;
    lost_count = 0.0;

elseif locked_visible
    locked_lon = locked_lon_now;
    locked_speed = locked_speed_now;
    lost_count = 0.0;

    if raw_id == locked_id || raw_id < 0.0
        pending_id = -1.0;
        pending_count = 0.0;
    elseif raw_lon < locked_lon_now-SWITCH_ADVANTAGE_M
        if pending_id == raw_id
            pending_count = pending_count+1.0;
        else
            pending_id = raw_id;
            pending_count = 1.0;
        end

        if pending_count >= PERSISTENCE_SAMPLES
            locked_id = raw_id;
            locked_lon = raw_lon;
            locked_speed = raw_speed;
            pending_id = -1.0;
            pending_count = 0.0;
            had_range = false;
        end
    else
        pending_id = -1.0;
        pending_count = 0.0;
    end
else
    lost_count = lost_count+1.0;

    if raw_id >= 0.0
        if pending_id == raw_id
            pending_count = pending_count+1.0;
        else
            pending_id = raw_id;
            pending_count = 1.0;
        end

        if pending_count >= PERSISTENCE_SAMPLES
            locked_id = raw_id;
            locked_lon = raw_lon;
            locked_speed = raw_speed;
            pending_id = -1.0;
            pending_count = 0.0;
            lost_count = 0.0;
            had_range = false;
        end
    elseif lost_count >= LOSS_GRACE_SAMPLES
        locked_id = -1.0;
        locked_lon = MAX_LOOK_AHEAD_M + ORIGIN_TO_BUMPER_M;
        locked_speed = max(double(ego_speed),0.0);
        pending_id = -1.0;
        pending_count = 0.0;
        lost_count = 0.0;
        had_range = false;
    end
end

if locked_id >= 0.0
    lead_actor_id = locked_id;
    lead_valid = 1.0;
    gap_m = max(locked_lon-ORIGIN_TO_BUMPER_M,0.0);

    if had_range
        raw_rate = (locked_lon-previous_lon)/TS;
        raw_rate = min(max(raw_rate,-MAX_ABS_RANGE_RATE_MPS), ...
                       MAX_ABS_RANGE_RATE_MPS);
        filtered_range_rate = filtered_range_rate + ...
            RANGE_RATE_ALPHA*(raw_rate-filtered_range_rate);
    else
        filtered_range_rate = locked_speed-max(double(ego_speed),0.0);
    end

    previous_lon = locked_lon;
    had_range = true;
    range_rate = filtered_range_rate;

    speed_from_range = max(double(ego_speed)+range_rate,0.0);
    if abs(locked_speed-speed_from_range) <= 4.0
        lead_speed = 0.65*locked_speed+0.35*speed_from_range;
    else
        lead_speed = speed_from_range;
    end
else
    lead_speed = max(double(ego_speed),0.0);
    gap_m = MAX_LOOK_AHEAD_M;
    lead_actor_id = -1.0;
    lead_valid = 0.0;
    range_rate = 0.0;
    had_range = false;
end
end
