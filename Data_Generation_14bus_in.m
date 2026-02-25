clc;
clear;
close all;

%% Load Case and System Data
mpc = loadcase('case14');
n = size(mpc.bus, 1);
num_gens = size(mpc.gen, 1);

% Extract cost coefficients from the standard MATPOWER format
%a = mpc.gencost(:, 5);
a = [0.043;0.025;0.025;0.02;0.03];
%b = mpc.gencost(:, 6);
b = [30;20;45;35;50];
%c = mpc.gencost(:, 7);
c = [0;0;0;0;0];
gen_buses = mpc.gen(:, 1);
% pmin and pmax
%pmin = mpc.gen(:, 6);
%pmax = mpc.gen(:, 7);
pmin = [2;2;2;2;2];
pmax = [150;100;300;250;200];

Pd_base = mpc.bus(:, 3);

Bdc = makeBdc(mpc);
B = full(Bdc);

%% Stackelberg Model Configuration
leader_gen_idx = 2; % Generator 2 is the leader
follower_gen_indices = [1; 3; 4; 5]; % Generators 1, 3, 4, 5 are followers
num_followers = length(follower_gen_indices);

n_steps_leader_search = 100;
leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), n_steps_leader_search);

% --- NEW: Define ranges for random follower costs ---
follower_a_base = a(follower_gen_indices)';
follower_b_base = b(follower_gen_indices)';
follower_c_base = c(follower_gen_indices)';

follower_cost_ranges = struct();
for i = 1:num_followers
    follower_cost_ranges.(['a' num2str(i)]) = [follower_a_base(i) * 0.8, follower_a_base(i) * 1.2];
    follower_cost_ranges.(['b' num2str(i)]) = [follower_b_base(i) * 0.8, follower_b_base(i) * 1.2];
    follower_cost_ranges.(['c' num2str(i)]) = [follower_c_base(i) * 0.8, follower_c_base(i) * 1.2];
end

%% Data Generation for Neural Network Training
num_scenarios = 10000;

% Input X:
% 1. Full load vector (n)
% 2. Leader's generation (1)
% 3. Follower cost coefficients (3 * num_followers)
num_X_cols = n + 1 + 3 * num_followers;
X_nn_input = zeros(num_scenarios, num_X_cols);

% Output Y:
% 1. LMP at leader bus (1)
% 2. Follower generations (num_followers)
num_Y_cols = 1 + num_followers;
Y_nn_outputs = zeros(num_scenarios, num_Y_cols);

fprintf('Generating %d optimal Stackelberg equilibrium points with variable follower costs...\n', num_scenarios);
scenario_count = 0;
attempt_count = 0;

while scenario_count < num_scenarios
    attempt_count = attempt_count + 1;

    % --- NEW: Randomize follower costs for the current scenario ---
    current_a_followers = zeros(1, num_followers);
    current_b_followers = zeros(1, num_followers);
    current_c_followers = zeros(1, num_followers);
    for i = 1:num_followers
        a_range = follower_cost_ranges.(['a' num2str(i)]);
        b_range = follower_cost_ranges.(['b' num2str(i)]);
        c_range = follower_cost_ranges.(['c' num2str(i)]);
        current_a_followers(i) = a_range(1) + (a_range(2) - a_range(1)) * rand();
        current_b_followers(i) = b_range(1) + (b_range(2) - b_range(1)) * rand();
        current_c_followers(i) = c_range(1) + (c_range(2) - c_range(1)) * rand();
    end
    
    a_scenario = a;
    b_scenario = b;
    c_scenario = c;
    a_scenario(follower_gen_indices) = current_a_followers';
    b_scenario(follower_gen_indices) = current_b_followers';
    c_scenario(follower_gen_indices) = current_c_followers';
    
    % --- Randomize load (using the multiplier method from previous code) ---
    a_mul = 3.37;
    b_mul = 3.86;
    mul = a_mul + (b_mul - a_mul) * rand();
    current_Pd = Pd_base * mul;
    
    leader_profits_k = zeros(n_steps_leader_search, 1);
    follower_pg_results_k = zeros(n_steps_leader_search, num_followers);
    lmp_at_leader_bus_k_all = zeros(n_steps_leader_search, 1);

    for k = 1:n_steps_leader_search
        pg_leader_candidate = leader_pg_range(k);

        % --- Solve Follower's Problem ---
        n_vars_f = num_followers + n;
        
        % --- IMPORTANT CHANGE: Use randomized costs here! ---
        H_f = diag([2 * a_scenario(follower_gen_indices)', zeros(1, n)]);
        f_f = [b_scenario(follower_gen_indices)', zeros(1, n)]';

        lb_f = [pmin(follower_gen_indices)', -inf(1, n)]';
        ub_f = [pmax(follower_gen_indices)', inf(1, n)]';
        
        Aeq_f = zeros(n, n_vars_f);
        beq_f = current_Pd;
        leader_bus = gen_buses(leader_gen_idx);
        beq_f(leader_bus) = beq_f(leader_bus) - pg_leader_candidate;

        for i = 1:n
            for j = 1:num_followers
                follower_bus = gen_buses(follower_gen_indices(j));
                if i == follower_bus
                    Aeq_f(i, j) = 1;
                end
            end
            Aeq_f(i, num_followers+1:end) = -B(i, :);
        end

        ref_bus_idx = find(mpc.bus(:, 2) == 3, 1);
        if isempty(ref_bus_idx)
            ref_bus_idx = 1;
        end
        ref_delta_idx_f = num_followers + ref_bus_idx;
        Aeq_f = [Aeq_f; zeros(1, n_vars_f)];
        Aeq_f(end, ref_delta_idx_f) = 1;
        beq_f = [beq_f; 0];
        
        options = optimoptions('quadprog', 'Display', 'off');
        [x_f_opt, ~, exitflag, ~, lambda] = quadprog(H_f, f_f, [], [], Aeq_f, beq_f, lb_f, ub_f, [], options);

        if exitflag > 0
            pg_followers_k = x_f_opt(1:num_followers);
            lmp_at_leader_bus_k_temp = lambda.eqlin(leader_bus);
            
            % --- IMPORTANT CHANGE: Use randomized costs here! ---
            leader_cost = a_scenario(leader_gen_idx)*pg_leader_candidate^2 + b_scenario(leader_gen_idx)*pg_leader_candidate + c_scenario(leader_gen_idx);
            leader_revenue = pg_leader_candidate * lmp_at_leader_bus_k_temp;
            leader_profits_k(k) = leader_revenue - leader_cost;
            
            follower_pg_results_k(k, :) = pg_followers_k';
            lmp_at_leader_bus_k_all(k) = lmp_at_leader_bus_k_temp;
        else
            leader_profits_k(k) = -inf;
        end
    end
    
    [max_profit_scenario, best_idx_scenario] = max(leader_profits_k);
    
    if max_profit_scenario > -inf && isfinite(max_profit_scenario)
        optimal_pg_leader_scenario = leader_pg_range(best_idx_scenario);
        optimal_pg_followers_scenario = follower_pg_results_k(best_idx_scenario, :)';
        optimal_lmp_leader_scenario = lmp_at_leader_bus_k_all(best_idx_scenario);
        
        % --- IMPORTANT CHANGE: Store random costs in the input vector ---
        X_nn_input(scenario_count + 1, :) = [current_Pd', optimal_pg_leader_scenario, current_a_followers, current_b_followers, current_c_followers];
        Y_nn_outputs(scenario_count + 1, :) = [optimal_lmp_leader_scenario, optimal_pg_followers_scenario'];
        
        scenario_count = scenario_count + 1;
        
        if mod(scenario_count, 1000) == 0
            fprintf('... %d optimal scenarios generated (attempts: %d)\n', scenario_count, attempt_count);
        end
    end
end

fprintf('All %d optimal scenarios generated successfully in %d attempts.\n', scenario_count, attempt_count);
save('stackelberg_optimal_points_variable_costs_dataset.mat', 'X_nn_input', 'Y_nn_outputs');
fprintf('Dataset saved to stackelberg_optimal_points_variable_costs_dataset.mat\n');