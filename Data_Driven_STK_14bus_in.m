clc;
clear;
close all;

%% 1. Load Data, Trained Network, and System Case
% Load normalization parameters from the training data.
% We assume the network was trained on variable costs, so we load the corresponding dataset for normalization.
load('stackelberg_optimal_points_variable_costs_dataset.mat', 'X_nn_input', 'Y_nn_outputs');
[~, ps_x] = mapminmax(X_nn_input', 0, 1);
[~, ps_y] = mapminmax(Y_nn_outputs', 0, 1);
%%
% Load the trained neural network.
% Note: The network should be trained on the variable costs dataset.
load('trainedNet_14bus_optimized.mat', 'net', 'ps_x', 'ps_y');

% Load the power system case data.
mpc = loadcase('case14');

% Define the specific load vector for the current online operation scenario.
online_demand_vector = mpc.bus(:, 3) * 3.85;
online_Pd_total = sum(online_demand_vector);

%% 2. Stackelberg Configuration for Case14
leader_gen_idx = 2; 
follower_gen_indices = [1; 3; 4; 5];
num_followers = length(follower_gen_indices);

% Extract generator parameters.
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

leader_bus = mpc.gen(leader_gen_idx, 1);
a1 = a(leader_gen_idx); b1 = b(leader_gen_idx); c1 = c(leader_gen_idx);

% Define uncertainty ranges for followers (based on a +/- 20% rule).
follower_a_base = a(follower_gen_indices)';
follower_b_base = b(follower_gen_indices)';
follower_c_base = c(follower_gen_indices)';

follower_cost_ranges = struct();
for i = 1:num_followers
    follower_cost_ranges.(['a' num2str(i)]) = [follower_a_base(i) * 0.8, follower_a_base(i) * 1.2];
    follower_cost_ranges.(['b' num2str(i)]) = [follower_b_base(i) * 0.8, follower_b_base(i) * 1.2];
    follower_cost_ranges.(['c' num2str(i)]) = [follower_c_base(i) * 0.8, follower_c_base(i) * 1.2];
end

num_uncertainty_samples = 100;

% Define tolerances similar to your complete information code.
tolerance = 0.1;
gen_limit_tolerance = 0.05;

% Define the search range for the leader's generation using linspace.
leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), 200);

%% 3. Maximin Profit Search
Pg_leader_list = [];
Maximin_Profit1_results = [];

fprintf('Starting online Stackelberg search for an optimal leader strategy using Maximin Profit criterion...\n');
fprintf('Current Total Demand: %.2f MW\n', online_Pd_total);

for pg_leader_candidate = leader_pg_range
    
    profit1_for_this_P1_across_scenarios = zeros(num_uncertainty_samples, 1);

    for s = 1:num_uncertainty_samples
        % Randomize follower costs.
        current_a_followers = zeros(num_followers, 1);
        current_b_followers = zeros(num_followers, 1);
        current_c_followers = zeros(num_followers, 1);
        for i = 1:num_followers
            a_range = follower_cost_ranges.(['a' num2str(i)]);
            b_range = follower_cost_ranges.(['b' num2str(i)]);
            c_range = follower_cost_ranges.(['c' num2str(i)]);
            current_a_followers(i) = a_range(1) + (a_range(2) - a_range(1)) * rand();
            current_b_followers(i) = b_range(1) + (b_range(2) - b_range(1)) * rand();
            current_c_followers(i) = c_range(1) + (c_range(2) - c_range(1)) * rand();
        end

        % Construct the neural network input vector with the FULL demand vector.
        input_vector = [online_demand_vector;
                        pg_leader_candidate;
                        current_a_followers;
                        current_b_followers;
                        current_c_followers];
        
        input_norm = mapminmax('apply', input_vector, ps_x);
        Y_pred_norm = net(input_norm);
        Y_pred = mapminmax('reverse', Y_pred_norm, ps_y);

        % Extract predicted outputs.
        lmp_pred = Y_pred(1);
        pg_followers_pred = Y_pred(2:end);
        
        % Check feasibility using the defined tolerances.
        is_followers_valid = all(pg_followers_pred >= pmin(follower_gen_indices) - gen_limit_tolerance) && ...
                             all(pg_followers_pred <= pmax(follower_gen_indices) + gen_limit_tolerance);
        
        total_generation = pg_leader_candidate + sum(pg_followers_pred);
        power_balance_is_good = (abs(total_generation - online_Pd_total) <= tolerance);

        % Calculate profit if feasible.
        if is_followers_valid && power_balance_is_good
            leader_cost = a1 * pg_leader_candidate^2 + b1 * pg_leader_candidate + c1;
            leader_profit = (lmp_pred * pg_leader_candidate) - leader_cost;
            profit1_for_this_P1_across_scenarios(s) = leader_profit;
        else
            profit1_for_this_P1_across_scenarios(s) = -inf;
        end
    end
    
    feasible_profits = profit1_for_this_P1_across_scenarios(profit1_for_this_P1_across_scenarios > -inf);

    if ~isempty(feasible_profits)
        min_profit_for_P1 = min(feasible_profits);
        Pg_leader_list = [Pg_leader_list; pg_leader_candidate];
        Maximin_Profit1_results = [Maximin_Profit1_results; min_profit_for_P1];
    end
end

fprintf('Maximin search finished.\n\n');

%% 4. Display Optimal Results
if isempty(Pg_leader_list)
    disp('No feasible solution found.');
else
    [max_of_min_profit, idx_best] = max(Maximin_Profit1_results);
    best_Pg_leader = Pg_leader_list(idx_best);
    
    % Use nominal costs to generate the final report
    nominal_input = [online_demand_vector;
                     best_Pg_leader;
                     follower_a_base'; follower_b_base'; follower_c_base'];
                    
    input_norm_opt_report = mapminmax('apply', nominal_input, ps_x);
    Y_pred_norm_opt_report = net(input_norm_opt_report);
    Y_pred_opt_report = mapminmax('reverse', Y_pred_norm_opt_report, ps_y);
    
    lmp_pred = Y_pred_opt_report(1);
    pg_followers_pred = Y_pred_opt_report(2:end);
    
    % --- BEGINNING OF MODIFICATIONS ---
    
    % Calculate leader's cost (already done in your code, just re-calculating for clarity)
    leader_cost = a1 * best_Pg_leader^2 + b1 * best_Pg_leader + c1;
    
    % Calculate followers' costs based on nominal coefficients
    follower_costs = zeros(num_followers, 1);
    for i = 1:num_followers
        pg_i = pg_followers_pred(i);
        a_i = follower_a_base(i);
        b_i = follower_b_base(i);
        c_i = follower_c_base(i);
        follower_costs(i) = a_i * pg_i^2 + b_i * pg_i + c_i;
    end
    
    % Calculate total system cost
    total_system_cost = leader_cost + sum(follower_costs);

    % --- END OF MODIFICATIONS ---

    leader_profit = (lmp_pred * best_Pg_leader) - leader_cost;
    
    fprintf('\n---------------------------------\n');
    fprintf('Optimal Leader Power (Gen 2): %.2f MW\n', best_Pg_leader);
    fprintf('Optimal Follower Powers (based on nominal costs):\n');
    for i = 1:num_followers
        fprintf('  Gen %d: %.2f MW\n', follower_gen_indices(i), pg_followers_pred(i));
    end
    fprintf('---------------------------------\n');
    fprintf('Total Generation: %.2f MW\n', best_Pg_leader + sum(pg_followers_pred));
    fprintf('Total Demand:     %.2f MW\n', online_Pd_total);
    fprintf('---------------------------------\n');
    fprintf('Maximin Leader Profit: %.2f $\n', max_of_min_profit);

    % --- BEGINNING OF MODIFICATIONS ---

    fprintf('Leader Generation Cost: %.2f $\n', leader_cost);
    fprintf('Follower Generation Costs:\n');
    for i = 1:num_followers
        fprintf('  Gen %d Cost: %.2f $\n', follower_gen_indices(i), follower_costs(i));
    end
    fprintf('Total System Generation Cost: %.2f $\n', total_system_cost);

    % --- END OF MODIFICATIONS ---
    
    fprintf('---------------------------------\n');
end