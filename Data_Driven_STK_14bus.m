clc;
clear;
close all;

%% Load Data and Trained Network
load('stackelberg_optimal_points_dataset.mat', 'X_nn_input', 'Y_nn_outputs');
[~, ps_x] = mapminmax(X_nn_input', 0, 1);
[~, ps_y] = mapminmax(Y_nn_outputs', 0, 1);

load('trainedNet_14bus_optimized.mat','net', 'ps_x', 'ps_y');

%% Load Case Data
mpc = loadcase('case14');
num_gens = size(mpc.gen, 1);
gen_buses = mpc.gen(:, 1);

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

leader_gen_idx = 2;
follower_gen_indices = [1; 3; 4; 5];
num_followers = length(follower_gen_indices);

leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), 200);

% Define the full load demand vector for online operation
online_demand_vector = mpc.bus(:, 3) * 3.45;

%% Search for Optimal Leader Strategy
tolerance = 0.1;
gen_limit_tolerance = 0.05;

Pg_leader_list = [];
profit_leader_list = [];
Pg_followers_list = [];

fprintf('Starting online Stackelberg search for an optimal leader strategy...\n');

for pg_leader_candidate = leader_pg_range
    follower_a = a(follower_gen_indices);
    follower_b = b(follower_gen_indices);
    follower_c = c(follower_gen_indices);
    
    % Construct the neural network input vector with the full demand vector
    input_vector = [online_demand_vector;
                    pg_leader_candidate;
                    follower_a;
                    follower_b;
                    follower_c];
    
    input_norm = mapminmax('apply', input_vector, ps_x);
    
    Y_pred_norm = net(input_norm);
    
    Y_pred = mapminmax('reverse', Y_pred_norm, ps_y);
    
    lmp_pred = Y_pred(1);
    pg_followers_pred = Y_pred(2:end);
    
    is_followers_valid = all(pg_followers_pred >= pmin(follower_gen_indices) - gen_limit_tolerance) && ...
                         all(pg_followers_pred <= pmax(follower_gen_indices) + gen_limit_tolerance);
    
    total_generation = pg_leader_candidate + sum(pg_followers_pred);
    power_balance_is_good = (abs(total_generation - sum(online_demand_vector)) <= tolerance);
    
    if is_followers_valid && power_balance_is_good
        leader_cost = a(leader_gen_idx) * pg_leader_candidate^2 + ...
                      b(leader_gen_idx) * pg_leader_candidate + ...
                      c(leader_gen_idx);
        leader_profit = (lmp_pred * pg_leader_candidate) - leader_cost;
        
        Pg_leader_list = [Pg_leader_list; pg_leader_candidate];
        profit_leader_list = [profit_leader_list; leader_profit];
        Pg_followers_list = [Pg_followers_list; pg_followers_pred'];
    end
end

%% Display Optimal Results
if isempty(Pg_leader_list)
    disp('No feasible solution found.');
else
    [max_profit, idx_best] = max(profit_leader_list);
    
    best_Pg_leader = Pg_leader_list(idx_best);
    best_Pg_followers = Pg_followers_list(idx_best, :);
    
    % --- Start of the user-requested changes ---
    
    leader_cost = a(leader_gen_idx) * best_Pg_leader^2 + ...
                  b(leader_gen_idx) * best_Pg_leader + ...
                  c(leader_gen_idx);
    
    % ?????? ????? ????? ????????
    follower_costs = zeros(num_followers, 1);
    for i = 1:num_followers
        follower_costs(i) = a(follower_gen_indices(i)) * best_Pg_followers(i)^2 + ...
                            b(follower_gen_indices(i)) * best_Pg_followers(i) + ...
                            c(follower_gen_indices(i));
    end
    
    % ?????? ????? ?? ?????
    total_cost = leader_cost + sum(follower_costs);
    
    % --- End of the user-requested changes ---
    
    fprintf('\n---------------------------------\n');
    fprintf('Optimal Leader Power (Gen 2): %.2f MW\n', best_Pg_leader);
    fprintf('Optimal Follower Powers:\n');
    for i = 1:num_followers
        fprintf('  Gen %d: %.2f MW\n', follower_gen_indices(i), best_Pg_followers(i));
    end
    fprintf('---------------------------------\n');
    fprintf('Total Generation: %.2f MW\n', best_Pg_leader + sum(best_Pg_followers));
    fprintf('Total Demand: %.2f MW\n', sum(online_demand_vector));
    fprintf('Max Leader Profit: %.2f $\n', max_profit);
    fprintf('---------------------------------\n');

    % --- Start of the new output display ---

    fprintf('Optimal Leader Cost: %.2f $\n', leader_cost);
    fprintf('Optimal Follower Costs:\n');
    for i = 1:num_followers
        fprintf('  Gen %d: %.2f $\n', follower_gen_indices(i), follower_costs(i));
    end
    fprintf('Total System Cost: %.2f $\n', total_cost);
    fprintf('---------------------------------\n');
end