function packet_index = packet_counter(sim_time, sample_time)
%#codegen
%PACKET_COUNTER Deterministic packet counter reset by simulation time.

Ts = max(double(sample_time),1.0e-3);
packet_index = floor(double(sim_time)/Ts + 1.0e-7);
end
