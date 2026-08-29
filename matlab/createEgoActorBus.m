function createEgoActorBus()
%CREATEEGOACTORBUS Defines the 7-field pose bus the Scenario Reader ego
% input port expects: [ActorID, Position, Velocity, Roll, Pitch, Yaw, AngularVelocity]
% Angles are in DEGREES, angular velocity in deg/s (drivingScenario convention).

    e = repmat(Simulink.BusElement, 7, 1);
    names = {'ActorID','Position','Velocity','Roll','Pitch','Yaw','AngularVelocity'};
    dims  = { 1,      [1 3],     [1 3],     1,     1,      1,    [1 3] };
    for k = 1:7
        e(k).Name       = names{k};
        e(k).Dimensions = dims{k};
        e(k).DataType   = 'double';
    end
    EgoActorBus = Simulink.Bus;    
    EgoActorBus.Elements = e;
    assignin('base', 'EgoActorBus', EgoActorBus);
end