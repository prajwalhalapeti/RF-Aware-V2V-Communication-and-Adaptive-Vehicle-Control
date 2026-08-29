function [ego_bus, path_s, path_progress, path_complete] = ...
    speed_to_ego_bus_final(v, sim_time, egoPathX, egoPathY, egoPathS)
%#codegen
%SPEED_TO_EGO_BUS_FINAL Path-locked ego pose with deterministic reset.

TS = 0.1;

persistent sCurrent previousYaw previousTime

if isempty(previousTime)
    sCurrent = 0.0;
    previousYaw = 0.0;
    previousTime = -1.0;
end

if sim_time < previousTime || ...
        (sim_time <= 1.0e-9 && previousTime > 0.0)
    sCurrent = 0.0;
    previousYaw = 0.0;
end
previousTime = sim_time;

pathLength = egoPathS(end);
speed = min(max(double(v),0.0),20.0);
sCurrent = min(sCurrent+speed*TS,pathLength);

x = interp1(egoPathS,egoPathX,sCurrent,'linear');
y = interp1(egoPathS,egoPathY,sCurrent,'linear');

ds = 0.5;
sA = max(sCurrent-ds,0.0);
sB = min(sCurrent+ds,pathLength);

xA = interp1(egoPathS,egoPathX,sA,'linear');
yA = interp1(egoPathS,egoPathY,sA,'linear');
xB = interp1(egoPathS,egoPathX,sB,'linear');
yB = interp1(egoPathS,egoPathY,sB,'linear');

dx = xB-xA;
dy = yB-yA;

if dx == 0.0 && dy == 0.0
    yawDeg = previousYaw;
else
    yawDeg = atan2(dy,dx)*180.0/pi;
end

deltaYaw = yawDeg-previousYaw;
if deltaYaw > 180.0
    deltaYaw = deltaYaw-360.0;
elseif deltaYaw < -180.0
    deltaYaw = deltaYaw+360.0;
end
yawRate = deltaYaw/TS;
previousYaw = yawDeg;

yawRad = yawDeg*pi/180.0;

ego_bus.ActorID = double(9);
ego_bus.Position = [x,y,0.0];
ego_bus.Velocity = [speed*cos(yawRad),speed*sin(yawRad),0.0];
ego_bus.Roll = 0.0;
ego_bus.Pitch = 0.0;
ego_bus.Yaw = yawDeg;
ego_bus.AngularVelocity = [0.0,0.0,yawRate];

path_s = sCurrent;
path_progress = sCurrent/max(pathLength,eps);
path_complete = double(sCurrent >= pathLength-0.25);
end
