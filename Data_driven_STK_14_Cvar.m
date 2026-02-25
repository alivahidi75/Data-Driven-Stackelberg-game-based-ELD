clc;
clear;
close all;

%% 1. Load Data, Trained Network, and System Case
load('stackelberg_optimal_points_variable_costs_dataset.mat', 'X_nn_input', 'Y_nn_outputs');
[~, ps_x] = mapminmax(X_nn_input', 0, 1);
[~, ps_y] = mapminmax(Y_nn_outputs', 0, 1);

load('trainedNet_14bus_optimized.mat', 'net', 'ps_x', 'ps_y');
mpc = loadcase('case14');

online_demand_vector = mpc.bus(:, 3) * 2;
online_Pd_total = sum(online_demand_vector);

%% 2. Stackelberg Configuration
leader_gen_idx = 2; 
follower_gen_indices = [1; 3; 4; 5];
num_followers = length(follower_gen_indices);

a = [0.043;0.025;0.025;0.02;0.03];
b = [30;20;45;35;50];
c = [0;0;0;0;0];

pmin = [2;2;2;2;2];
pmax = [150;100;300;250;200];

a1 = a(leader_gen_idx); 
b1 = b(leader_gen_idx); 
c1 = c(leader_gen_idx);

follower_a_base = a(follower_gen_indices)';
follower_b_base = b(follower_gen_indices)';
follower_c_base = c(follower_gen_indices)';

follower_cost_ranges = struct();
for i = 1:num_followers
    follower_cost_ranges.(['a' num2str(i)]) = [follower_a_base(i)*0.8, follower_a_base(i)*1.2];
    follower_cost_ranges.(['b' num2str(i)]) = [follower_b_base(i)*0.8, follower_b_base(i)*1.2];
    follower_cost_ranges.(['c' num2str(i)]) = [follower_c_base(i)*0.8, follower_c_base(i)*1.2];
end

num_uncertainty_samples = 100;
tolerance = 0.1;
gen_limit_tolerance = 0.05;

leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), 50);

%% CVaR Parameter
alpha = 0.95;

%% 3. CVaR-Based Search
Pg_leader_list = [];
CVaR_results = [];

fprintf('Starting online Stackelberg search using CVaR criterion...\n');
fprintf('Current Total Demand: %.2f MW\n', online_Pd_total);

for pg_leader_candidate = leader_pg_range
    
    profit_scenarios = zeros(num_uncertainty_samples,1);

    for s = 1:num_uncertainty_samples
        
        current_a = zeros(num_followers,1);
        current_b = zeros(num_followers,1);
        current_c = zeros(num_followers,1);
        
        for i = 1:num_followers
            a_range = follower_cost_ranges.(['a' num2str(i)]);
            b_range = follower_cost_ranges.(['b' num2str(i)]);
            c_range = follower_cost_ranges.(['c' num2str(i)]);
            
            current_a(i) = a_range(1) + (a_range(2)-a_range(1))*rand();
            current_b(i) = b_range(1) + (b_range(2)-b_range(1))*rand();
            current_c(i) = c_range(1) + (c_range(2)-c_range(1))*rand();
        end
        
        input_vector = [online_demand_vector;
                        pg_leader_candidate;
                        current_a;
                        current_b;
                        current_c];
        
        input_norm = mapminmax('apply', input_vector, ps_x);
        Y_pred_norm = net(input_norm);
        Y_pred = mapminmax('reverse', Y_pred_norm, ps_y);
        
        lmp_pred = Y_pred(1);
        pg_followers_pred = Y_pred(2:end);
        
        is_valid = all(pg_followers_pred >= pmin(follower_gen_indices)-gen_limit_tolerance) && ...
                   all(pg_followers_pred <= pmax(follower_gen_indices)+gen_limit_tolerance);
        
        total_generation = pg_leader_candidate + sum(pg_followers_pred);
        balance_ok = abs(total_generation - online_Pd_total) <= tolerance;
        
        if is_valid && balance_ok
            leader_cost = a1*pg_leader_candidate^2 + b1*pg_leader_candidate + c1;
            leader_profit = (lmp_pred*pg_leader_candidate) - leader_cost;
            profit_scenarios(s) = leader_profit;
        else
            profit_scenarios(s) = -inf;
        end
    end
    
    feasible_profits = profit_scenarios(profit_scenarios > -inf);
    
    if ~isempty(feasible_profits)
        sorted_profits = sort(feasible_profits);
        k = max(1, ceil((1-alpha)*length(sorted_profits)));
        cvar_value = mean(sorted_profits(1:k));
        
        Pg_leader_list = [Pg_leader_list; pg_leader_candidate];
        CVaR_results = [CVaR_results; cvar_value];
    end
end

fprintf('CVaR search finished.\n\n');

%% 4. Display Optimal Results
if isempty(Pg_leader_list)
    disp('No feasible solution found.');
else
    
    [best_cvar, idx_best] = max(CVaR_results);
    best_Pg_leader = Pg_leader_list(idx_best);
    
    nominal_input = [online_demand_vector;
                     best_Pg_leader;
                     follower_a_base'; 
                     follower_b_base'; 
                     follower_c_base'];
                 
    input_norm = mapminmax('apply', nominal_input, ps_x);
    Y_pred_norm = net(input_norm);
    Y_pred_opt_report = mapminmax('reverse', Y_pred_norm, ps_y);
    
    lmp_pred = Y_pred_opt_report(1);
    pg_followers_pred = Y_pred_opt_report(2:end);
    
    leader_cost = a1*best_Pg_leader^2 + b1*best_Pg_leader + c1;
    
    follower_costs = zeros(num_followers, 1);
    for i = 1:num_followers
        pg_i = pg_followers_pred(i);
        a_i = follower_a_base(i);
        b_i = follower_b_base(i);
        c_i = follower_c_base(i);
        follower_costs(i) = a_i*pg_i^2 + b_i*pg_i + c_i;
    end
    
    total_system_cost = leader_cost + sum(follower_costs);
    leader_profit = (lmp_pred*best_Pg_leader) - leader_cost;
    
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
    fprintf('CVaR(%.2f) Leader Profit: %.2f $\n', alpha, best_cvar);
    
    fprintf('Leader Generation Cost: %.2f $\n', leader_cost);
    fprintf('Follower Generation Costs:\n');
    
    for i = 1:num_followers
        fprintf('  Gen %d Cost: %.2f $\n', follower_gen_indices(i), follower_costs(i));
    end
    
    fprintf('Total System Generation Cost: %.2f $\n', total_system_cost);
    fprintf('---------------------------------\n');
end
