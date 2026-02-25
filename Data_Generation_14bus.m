clc;
clear;
close all;

%% Load Case and System Data
mpc = loadcase('case14');
n = size(mpc.bus, 1);
num_gens = size(mpc.gen, 1);

a = [0.043;0.025;0.025;0.02;0.03];
b = [30;20;45;35;50];
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
leader_gen_idx = 2;
follower_gen_indices = [1; 3; 4; 5];
num_followers = length(follower_gen_indices);

n_steps_leader_search = 100;
leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), n_steps_leader_search);

%% Generate Load Scenarios
num_scenarios = 1000;
a_mul = 3.37;
b_mul = 3.86;
mul = a_mul + (b_mul - a_mul) * rand(1, num_scenarios);
Pd_scenarios = Pd_base * mul;

%% Data Generation for Neural Network Training
% Extract follower cost coefficients
follower_a = a(follower_gen_indices)';
follower_b = b(follower_gen_indices)';
follower_c = c(follower_gen_indices)';

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

fprintf('Generating %d optimal Stackelberg equilibrium points...\n', num_scenarios);
scenario_count = 0;

while scenario_count < num_scenarios
    
    current_Pd = Pd_scenarios(:, scenario_count + 1);
    
    leader_profits_k = zeros(n_steps_leader_search, 1);
    follower_pg_results_k = zeros(n_steps_leader_search, num_followers);
    lmp_at_leader_bus_k_all = zeros(n_steps_leader_search, 1);

    for k = 1:n_steps_leader_search
        pg_leader_candidate = leader_pg_range(k);

        % --- Solve Follower's Problem ---
        n_vars_f = num_followers + n;
        H_f = diag([2 * a(follower_gen_indices)', zeros(1, n)]);
        f_f = [b(follower_gen_indices)', zeros(1, n)]';

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
            
            leader_cost = a(leader_gen_idx)*pg_leader_candidate^2 + b(leader_gen_idx)*pg_leader_candidate + c(leader_gen_idx);
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
        
        % Store only the optimal point for this scenario
        X_nn_input(scenario_count + 1, :) = [current_Pd', optimal_pg_leader_scenario, follower_a, follower_b, follower_c];
        Y_nn_outputs(scenario_count + 1, :) = [optimal_lmp_leader_scenario, optimal_pg_followers_scenario'];
        
        scenario_count = scenario_count + 1;
        
        if mod(scenario_count, 100) == 0
            fprintf('... %d optimal scenarios generated\n', scenario_count);
        end
    end
end

fprintf('All %d optimal scenarios generated successfully.\n', scenario_count);
save('stackelberg_optimal_points_dataset.mat', 'X_nn_input', 'Y_nn_outputs');
fprintf('Dataset saved to stackelberg_optimal_points_dataset.mat\n');