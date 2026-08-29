function raw = v2v_zmq_request_experiment(packet_index, sim_time, ...
    attack_code, attack_active, severity, run_seed, expected_device_id)
%V2V_ZMQ_REQUEST_EXPERIMENT Robust MATLAB -> Python ZeroMQ bridge.
% Compatible with rf_metrics_receiver_experiment and infer_service_experiments.py.

persistent context socket errorCount

raw = '';
if isempty(errorCount)
    errorCount = 0;
end

try
    zmq = py.importlib.import_module('zmq');

    if isempty(context)
        context = zmq.Context.instance();
    end

    if isempty(socket)
        socket = context.socket(zmq.REQ);
        socket.setsockopt(zmq.LINGER, int32(0));
        socket.setsockopt(zmq.SNDTIMEO, int32(5000));
        socket.setsockopt(zmq.RCVTIMEO, int32(5000));
        socket.connect('tcp://127.0.0.1:5557');
    end

    request = sprintf([ ...
        '{"packet_index":%d,"sim_time_s":%.9f,' ...
        '"attack_code":%d,"attack_active":%d,' ...
        '"severity":%.9f,"run_seed":%d,' ...
        '"expected_device_id":%d}'], ...
        int64(packet_index), double(sim_time), int32(attack_code), ...
        int32(attack_active), double(severity), int64(run_seed), ...
        int32(expected_device_id));

    socket.send_string(request);
    raw = char(socket.recv_string());

catch ME
    errorCount = errorCount + 1;
    if errorCount <= 5
        warning('v2v_zmq_request_experiment:BridgeFailure', ...
            'Experiment ZMQ request failed: %s', ME.message);
    end

    try
        if ~isempty(socket)
            socket.setsockopt(py.zmq.LINGER, int32(0));
            socket.close();
        end
    catch
    end
    socket = [];
    raw = '';
end
end
