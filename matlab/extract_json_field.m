function val = extract_json_field(json_bytes, field_name)
% extract_json_field  Extrinsic helper called by rf_metrics_receiver.
% Decodes raw ZMQ bytes and returns ONE named field as double.
% Lives as a standalone file so Simulink Coder treats the whole
% function as an opaque host call — no struct field resolution needed.
%
% USAGE (inside rf_metrics_receiver, declared extrinsic):
%   val = extract_json_field(raw_bytes, 'rf_quality');

s   = jsondecode(char(json_bytes));
val = double(s.(field_name));   % dynamic field access — fine in normal MATLAB
end