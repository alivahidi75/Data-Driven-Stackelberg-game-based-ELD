clc; 
clear; 
close all;

%% Load case (IEEE-14) 
mpc = loadcase('case14');
n = size(mpc.bus, 1);
num_gens = size(mpc.gen, 1);
gen_buses = mpc.gen(:, 1);

% Cost Coefficients (EXACTLY AS IN YOUR CODE) 
a = [0.043; 0.025; 0.025; 0.02; 0.03];
b = [30; 20; 45; 35; 50];
c = [0; 0; 0; 0; 0];

% Pmin / Pmax (EXACTLY AS IN YOUR CODE) 
pmin = [2; 2; 2; 2; 2];
pmax = [150; 100; 300; 250; 200];

% Fixed Base Load (Pd) 
Pd_base = mpc.bus(:, 3)*3.45; 
Pd_fixed = Pd_base; % Load is FIXED and NOMINAL in all scenarios

% --- B Matrix ---
B = makeBdc(mpc);
B = full(B);

%% --- Leader / Follower Setup ---
leader_gen_idx = 2;
follower_gen_indices = [1; 3; 4; 5];
num_followers = length(follower_gen_indices);

% --- Uncertainty Settings (ONLY FOLLOWER COSTS) ---
cost_uncertainty_factor = 0.2; % +/- 20% uncertainty range
follower_cost_ranges = struct();
follower_a_base = a(follower_gen_indices)';
follower_b_base = b(follower_gen_indices)';
follower_c_base = c(follower_gen_indices)';

for i = 1:num_followers
    val_a = follower_a_base(i);
    val_b = follower_b_base(i);
    val_c = follower_c_base(i);
    follower_cost_ranges.(['a' num2str(i)]) = [val_a*(1-cost_uncertainty_factor), val_a*(1+cost_uncertainty_factor)];
    follower_cost_ranges.(['b' num2str(i)]) = [val_b*(1-cost_uncertainty_factor), val_b*(1+cost_uncertainty_factor)];
    follower_cost_ranges.(['c' num2str(i)]) = [val_c*(1-cost_uncertainty_factor), val_c*(1+cost_uncertainty_factor)];
end

%% --- Simulation Parameters ---
n_steps = 100; % Same as your original code
leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), n_steps);
M = 100;       % Number of Monte Carlo scenarios per step
rng(0);        % Reproducibility

%% --- Pre-allocations ---
worst_profit = -inf(n_steps, 1);
worst_pg_followers = nan(n_steps, num_followers);
worst_lmp_leader = nan(n_steps, 1);

% To store scenario data for the ABSOLUTE worst case found
global_best_worst_profit = -inf;
best_worst_scenario_data = struct('a', cell(1,1), 'b', cell(1,1), 'c', cell(1,1)); % Pd is fixed, so only store costs

fprintf('Running Robust Stackelberg Simulation (Fixed Load, Cost Uncertainty, M=%d, n_steps=%d)...\n', M, n_steps);

%% --- Main Loop ---
for k = 1:n_steps
    PgL = leader_pg_range(k);
    
    profits_s = -inf(M, 1);
    pg_followers_s = nan(M, num_followers);
    lmp_leader_s = nan(M, 1);
    
    % Temporary storage to track worst scenario in this loop
    scenarios_data_step = struct('a', cell(M,1), 'b', cell(M,1), 'c', cell(M,1));

    for s = 1:M
        % 1. Load is FIXED: Pd = Pd_fixed; (No randomization here)
        
        % 2. Randomize Follower Costs
        a_rand = a; b_rand = b; c_rand = c;
        for i = 1:num_followers
            idx = follower_gen_indices(i);
            a_rng = follower_cost_ranges.(['a' num2str(i)]);
            b_rng = follower_cost_ranges.(['b' num2str(i)]);
            c_rng = follower_cost_ranges.(['c' num2str(i)]);
            a_rand(idx) = a_rng(1) + (a_rng(2)-a_rng(1))*rand();
            b_rand(idx) = b_rng(1) + (b_rng(2)-b_rng(1))*rand();
            c_rand(idx) = c_rng(1) + (c_rng(2)-c_rng(1))*rand();
        end
        
        % Store scenario data (only costs)
        scenarios_data_step(s).a = a_rand;
        scenarios_data_step(s).b = b_rand;
        scenarios_data_step(s).c = c_rand;

        % 3. Follower Optimization (QP)
        n_vars_f = num_followers + n;
        Hf = zeros(n_vars_f);
        Hf(1:num_followers, 1:num_followers) = diag(2 * a_rand(follower_gen_indices));
        ff = zeros(n_vars_f, 1);
        ff(1:num_followers) = b_rand(follower_gen_indices);
        
        lb_f = -inf(n_vars_f, 1); ub_f = inf(n_vars_f, 1);
        lb_f(1:num_followers) = pmin(follower_gen_indices);
        ub_f(1:num_followers) = pmax(follower_gen_indices);
        
        Aeq = zeros(n, n_vars_f);
        beq = Pd_fixed; % Use FIXED Load
        leader_bus = gen_buses(leader_gen_idx);
        beq(leader_bus) = beq(leader_bus) - PgL;
        
        for i = 1:n
            for j = 1:num_followers
                if i == gen_buses(follower_gen_indices(j))
                    Aeq(i,j) = 1;
                end
            end
            Aeq(i, num_followers+1:end) = -B(i,:);
        end
        
        ref_bus = 1; % Fixed ref bus 1
        Aeq = [Aeq; zeros(1,n_vars_f)];
        Aeq(end, num_followers + ref_bus) = 1;
        beq = [beq; 0];
        
        options = optimoptions('quadprog','Display','off');
        [x_f, ~, exitflag, ~, lambda] = quadprog(Hf, ff, [], [], Aeq, beq, lb_f, ub_f, [], options);
        
        if exitflag > 0
            pg_f = x_f(1:num_followers);
            lmp_curr = lambda.eqlin(leader_bus);
            
            % Note: Leader cost is calculated using randomized cost coeffs
            leader_cost = a_rand(leader_gen_idx)*PgL^2 + b_rand(leader_gen_idx)*PgL + c_rand(leader_gen_idx);
            leader_rev = PgL * lmp_curr;
            
            profits_s(s) = leader_rev - leader_cost;
            pg_followers_s(s,:) = pg_f';
            lmp_leader_s(s) = lmp_curr;
        else
            profits_s(s) = -inf; 
        end
    end
    
    % Find worst case for this PgL
    [min_profit_step, idx_min] = min(profits_s);
    worst_profit(k) = min_profit_step;
    
    if isfinite(min_profit_step)
        worst_pg_followers(k,:) = pg_followers_s(idx_min,:);
        worst_lmp_leader(k) = lmp_leader_s(idx_min);
        
        % Check if this is the BEST of the WORST so far (Maximin)
        if min_profit_step > global_best_worst_profit
             global_best_worst_profit = min_profit_step;
             best_worst_scenario_data.a = scenarios_data_step(idx_min).a;
             best_worst_scenario_data.b = scenarios_data_step(idx_min).b;
             best_worst_scenario_data.c = scenarios_data_step(idx_min).c;
        end
    end
    
    if mod(k, 10) == 0
        fprintf('Step %d/%d: PgL=%.2f, Worst Profit=%.2f\n', k, n_steps, PgL, worst_profit(k));
    end
end

fprintf('Simulation finished.\n\n');

%% --- Extract Optimal Results ---
[best_worst_profit, best_idx] = max(worst_profit);
best_PgL = leader_pg_range(best_idx);
best_pg_follow = worst_pg_followers(best_idx, :)';

% Construct Pg_opt
Pg_opt = zeros(num_gens, 1);
Pg_opt(leader_gen_idx) = best_PgL;
Pg_opt(follower_gen_indices) = best_pg_follow;

%% --- Re-Run QP for EXACT Reporting (Using Saved Worst-Case Cost Data) ---
% We need to re-run the QP one last time with the exact 'a', 'b', 'c' 
% from the identified worst-case scenario to get ALL LMPs and verify costs.

% Pd is FIXED, use Pd_fixed
Pd_worst = Pd_fixed; 
a_worst = best_worst_scenario_data.a;
b_worst = best_worst_scenario_data.b;
c_worst = best_worst_scenario_data.c;

pg_leader = best_PgL;

n_vars_f = num_followers + n;
Hf = zeros(n_vars_f);
Hf(1:num_followers, 1:num_followers) = diag(2 * a_worst(follower_gen_indices));
ff = zeros(n_vars_f, 1);
ff(1:num_followers) = b_worst(follower_gen_indices);
lb_f = -Inf(n_vars_f, 1); ub_f = Inf(n_vars_f, 1);
lb_f(1:num_followers) = pmin(follower_gen_indices);
ub_f(1:num_followers) = pmax(follower_gen_indices);

Aeq_f = zeros(n, n_vars_f);
beq_f = Pd_worst;
beq_f(leader_bus) = beq_f(leader_bus) - pg_leader;

for i = 1:n
    for j = 1:num_followers
        if i == gen_buses(follower_gen_indices(j))
            Aeq_f(i, j) = 1;
        end
    end
    Aeq_f(i, num_followers+1:end) = -B(i, :);
end
ref_bus = 1;
Aeq_f = [Aeq_f; zeros(1, n_vars_f)];
Aeq_f(end, num_followers + ref_bus) = 1;
beq_f = [beq_f; 0];

options = optimoptions('quadprog', 'Display', 'off');
[~, ~, ~, ~, lambda_final] = quadprog(Hf, ff, [], [], Aeq_f, beq_f, lb_f, ub_f, [], options);

% --- Final Calculations ---
LMP_values = lambda_final.eqlin(1:n); % Get LMPs for ALL buses

total_gen = sum(Pg_opt);
total_demand = sum(Pd_worst);
% Use worst-case costs (a_worst, b_worst, c_worst) for reporting
cost_per_gen = a_worst .* Pg_opt.^2 + b_worst .* Pg_opt + c_worst;
total_cost = sum(cost_per_gen);

profit_per_gen = zeros(num_gens, 1);
for i = 1:num_gens
    gen_bus_for_profit = gen_buses(i);
    profit_per_gen(i) = (Pg_opt(i) * LMP_values(gen_bus_for_profit)) - cost_per_gen(i);
end

%% --- Print Results (Original Format) ---
fprintf('---------------------------------------------------\n');
fprintf('--- Robust Stackelberg Equilibrium Results (Fixed Load) ---\n');
fprintf('---------------------------------------------------\n');
fprintf('Optimal Generator Outputs (Maximin Strategy):\n');
for i = 1:num_gens
    fprintf('  Pg%d (Bus %d) = %.2f MW\n', i, gen_buses(i), Pg_opt(i));
end
fprintf('---------------------------------------------------\n');
fprintf('Total Generation = %.2f MW\n', total_gen);
fprintf('Total Demand     = %.2f MW (Fixed Nominal Demand)\n', total_demand);
fprintf('Power Balance Error = %.4f MW\n', abs(total_gen - total_demand));
fprintf('---------------------------------------------------\n');
fprintf('Locational Marginal Prices (LMPs) - Worst Cost Scenario:\n');
for i = 1:n
    fprintf('  LMP at Bus %-2d = %.2f $/MWh\n', i, LMP_values(i));
end
fprintf('---------------------------------------------------\n');
fprintf('Profits per Generator (Worst Cost Scenario):\n');
for i = 1:num_gens
    fprintf('  Profit G%d = %.2f $\n', i, profit_per_gen(i));
end
fprintf('Costs per Generator (Worst Cost Scenario):\n');
for i = 1:num_gens
    fprintf('  Cost G%d = %.2f $\n', i, cost_per_gen(i));
end
fprintf('Total System Generation Cost = %.2f $\n', total_cost);
fprintf('---------------------------------------------------\n');

%% --- Plots ---
figure;
plot(leader_pg_range, worst_profit, 'b-', 'LineWidth', 2);
hold on;
plot(best_PgL, best_worst_profit, 'r*', 'MarkerSize', 10, 'LineWidth', 2);
title('Leader''s (Gen 2) Worst-Case Profit vs. Generation');
xlabel('Generation of Leader (Pg2) [MW]');
ylabel('Worst-Case Profit [$]');
legend('Profit Curve (Robust)', 'Maximin Point');
grid on;
box on;

figure;
bar_labels = cell(num_gens, 1);
for i = 1:num_gens
    bar_labels{i} = sprintf('G%d (Bus %d)', i, gen_buses(i));
end
bar(Pg_opt);
set(gca, 'xticklabel', bar_labels);
xtickangle(45);
title('Optimal Power Dispatch (Robust Equilibrium)');
ylabel('Power Output [MW]');
grid on;
box on;