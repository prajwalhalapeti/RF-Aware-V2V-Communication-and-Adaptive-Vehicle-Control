function allRuns = run_thesis_experiment_matrix()
%RUN_THESIS_EXPERIMENT_MATRIX Paired fixed-vs-adaptive regression.
%
% 10 seeds x 2 controllers x 4 conditions = 80 runs.
%
% Fixed and adaptive runs use identical seed, traffic scenario, attack
% window, I/Q selection, and packet-loss random numbers. Only
% ADAPTIVE_ENABLE changes.

mdl = 'v2v_urban_scenario';
load_system(mdl);

seeds = (1:10)';
controllers = [0 1]; % 0 fixed, 1 RF-adaptive
conditions = [0 1 2 3]; % clean, NLOS, jamming, spoofing

% A single main severity is used for the core 160-run matrix. Clean ignores
% severity. Add [0.35 0.65 1.0] later for a severity study.
mainSeverity = 0.75;

try
    set_param(mdl,'EnablePacing','off');
catch
end
set_param(mdl,'FastRestart','on');

rows = table;
runNumber = 0;
totalRuns = numel(seeds)*numel(controllers)*numel(conditions);

cleanup = onCleanup(@() set_param(mdl,'FastRestart','off'));

for s = 1:numel(seeds)
    seed = seeds(s);

    for c = 1:numel(conditions)
        condition = conditions(c);

        for a = 1:numel(controllers)
            adaptive = controllers(a);
            runNumber = runNumber+1;

            if condition == 0
                severity = 0.0;
            else
                severity = mainSeverity;
            end

            fprintf('Run %d/%d: seed=%d controller=%d condition=%d\n', ...
                runNumber,totalRuns,seed,adaptive,condition);

            in = Simulink.SimulationInput(mdl);
            in = in.setVariable('RUN_SEED',double(seed));
            in = in.setVariable('ADAPTIVE_ENABLE',double(adaptive));
            in = in.setVariable('CONDITION_CODE',double(condition));
            in = in.setVariable('ATTACK_SEVERITY',double(severity));
            in = in.setVariable('SPEED_LIMIT_MPS',15.0);
            in = in.setModelParameter('StopTime','120');

            simOut = sim(in);
            row = compute_run_metrics_v4( ...
                simOut,seed,adaptive,condition,severity);
            rows = [rows;row]; %#ok<AGROW>
        end
    end
end

allRuns = rows;

if ~exist('results','dir')
    mkdir('results');
end
writetable(allRuns,'results/thesis_all_runs.csv');

groupSummary = build_group_summary(allRuns);
writetable(groupSummary,'results/thesis_group_summary.csv');

paired = build_paired_differences(allRuns);
writetable(paired,'results/thesis_paired_differences.csv');

make_summary_figures(groupSummary);
end

function S = build_group_summary(T)
metrics = {'Collision','MinGap_m','MinTTC_s','HeadwayIncrease_s', ...
    'FalseBrakeTime_pct','PeakAbsJerk_mps3','P95AbsJerk_mps3', ...
    'DeviceMacroF1','NLOSF1','SILSuccess_pct','WirelessPDR_pct', ...
    'MeanAoI_s','MedianInference_ms','P95Inference_ms', ...
    'RFToHeadwayLatency_ms','RFToBrakeLatency_ms'};

rows = {};
controllers = unique(T.ControllerCode)';
conditions = unique(T.ConditionCode)';

for c = controllers
    for q = conditions
        mask = T.ControllerCode == c & T.ConditionCode == q;
        n = sum(mask);

        for m = 1:numel(metrics)
            name = metrics{m};
            x = T.(name)(mask);
            x = x(isfinite(x));

            if isempty(x)
                mu = NaN; sd = NaN; ci = NaN;
            else
                mu = mean(x);
                sd = std(x);
                if numel(x) > 1
                    ci = t_critical_975(numel(x)-1)*sd/sqrt(numel(x));
                else
                    ci = NaN;
                end
            end

            rows(end+1,:) = {c,q,string(name),n,mu,sd,ci}; %#ok<AGROW>
        end
    end
end

S = cell2table(rows,'VariableNames', ...
    {'ControllerCode','ConditionCode','Metric','N','Mean','SD','CI95HalfWidth'});
end

function value = t_critical_975(df)
if exist('tinv','file') == 2
    value = tinv(0.975,df);
    return;
end
% Two-sided 95%% Student-t critical values for df 1..30, followed by the
% normal approximation. The core experiment uses df=19.
tableValues = [12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 ...
    2.262 2.228 2.201 2.179 2.160 2.145 2.131 2.120 2.110 2.101 ...
    2.093 2.086 2.080 2.074 2.069 2.064 2.060 2.056 2.052 2.048 ...
    2.045 2.042];
if df >= 1 && df <= numel(tableValues)
    value = tableValues(df);
else
    value = 1.96;
end
end

function P = build_paired_differences(T)
metrics = {'Collision','MinGap_m','MinTTC_s','HeadwayIncrease_s', ...
    'FalseBrakeTime_pct','P95AbsJerk_mps3','WirelessPDR_pct', ...
    'MeanAoI_s','RFToHeadwayLatency_ms','RFToBrakeLatency_ms'};

rows = {};
conditions = unique(T.ConditionCode)';
seeds = unique(T.Seed)';

for q = conditions
    for s = seeds
        fixed = T(T.ConditionCode == q & T.Seed == s & ...
                  T.ControllerCode == 0,:);
        adaptive = T(T.ConditionCode == q & T.Seed == s & ...
                     T.ControllerCode == 1,:);

        if height(fixed) ~= 1 || height(adaptive) ~= 1
            continue;
        end

        for m = 1:numel(metrics)
            name = metrics{m};
            difference = adaptive.(name)-fixed.(name);
            rows(end+1,:) = {s,q,string(name),difference}; %#ok<AGROW>
        end
    end
end

P = cell2table(rows,'VariableNames', ...
    {'Seed','ConditionCode','Metric','AdaptiveMinusFixed'});
end

function make_summary_figures(S)
conditionNames = ["Clean","NLOS","Jamming","Spoofing"];
metrics = ["MinTTC_s","MinGap_m","HeadwayIncrease_s", ...
           "FalseBrakeTime_pct"];

for m = 1:numel(metrics)
    metric = metrics(m);
    fig = figure('Color','w','Units','inches','Position',[1 1 8 5]);
    means = nan(4,2);
    errors = nan(4,2);

    for q = 0:3
        for c = 0:1
            row = S(S.ConditionCode == q & S.ControllerCode == c & ...
                    S.Metric == metric,:);
            if height(row) == 1
                means(q+1,c+1) = row.Mean;
                errors(q+1,c+1) = row.CI95HalfWidth;
            end
        end
    end

    b = bar(means);
    hold on;
    for c = 1:2
        x = b(c).XEndPoints;
        errorbar(x,means(:,c),errors(:,c),'k.','LineWidth',1.0);
    end
    grid on;
    set(gca,'XTickLabel',conditionNames);
    legend('Fixed headway','RF-adaptive','Location','best');
    ylabel(strrep(metric,'_',' '));
    title(strrep(metric,'_',' '));

    exportgraphics(fig, ...
        fullfile('results',"comparison_"+metric+".png"), ...
        'Resolution',300);
    close(fig);
end
end
