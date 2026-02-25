clc;
clear;
close all;

%% 
mpc = loadcase('case14');
n = size(mpc.bus, 1);
num_gens = size(mpc.gen, 1);

%a = mpc.gencost(:, 5);
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

% Bus demand data (in MW)
Pd = mpc.bus(:, 3)*3.85; 

% Create DC Power Flow B matrix
B = makeBdc(mpc);
B = full(B);

%% 
leader_gen_idx = 2;
follower_gen_indices = [1; 3; 4; 5];
num_followers = length(follower_gen_indices);

n_steps = 100;
leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), n_steps);

leader_profits = zeros(n_steps, 1);
follower_pg_results = zeros(n_steps, num_followers);

fprintf('Running Stackelberg Simulation...\n');

%% 
for k = 1:n_steps
    pg_leader = leader_pg_range(k);

    n_vars_f = num_followers + n;

    H_f = zeros(n_vars_f);
    H_f(1:num_followers, 1:num_followers) = diag(2 * a(follower_gen_indices));
    f_f = zeros(n_vars_f, 1);
    f_f(1:num_followers) = b(follower_gen_indices);

    lb_f = -Inf(n_vars_f, 1);
    ub_f = Inf(n_vars_f, 1);
    lb_f(1:num_followers) = pmin(follower_gen_indices);
    ub_f(1:num_followers) = pmax(follower_gen_indices);
    
    Aeq_f = zeros(n, n_vars_f); 
    beq_f = Pd;

    leader_bus = gen_buses(leader_gen_idx);
    beq_f(leader_bus) = beq_f(leader_bus) - pg_leader;

    for i = 1:n 
        for j = 1:num_followers
            follower_bus = gen_buses(follower_gen_indices(j));
            if i == follower_bus
                Aeq_f(i, j) = 1; 
            end
        end
        Aeq_f(i, num_followers+1:end) = -B(i, :);
    end

    ref_bus = 1; 
    ref_delta_idx_f = num_followers + ref_bus;
    Aeq_f = [Aeq_f; zeros(1, n_vars_f)];
    Aeq_f(end, ref_delta_idx_f) = 1;
    beq_f = [beq_f; 0];

    options = optimoptions('quadprog', 'Display', 'off');
    [x_f_opt, ~, exitflag, ~, lambda] = quadprog(H_f, f_f, [], [], Aeq_f, beq_f, lb_f, ub_f, [], options);
    
    if exitflag > 0
        pg_followers = x_f_opt(1:num_followers);
        
        lmp_leader = lambda.eqlin(leader_bus); 

        leader_cost = a(leader_gen_idx) * pg_leader^2 + b(leader_gen_idx) * pg_leader + c(leader_gen_idx);
        leader_revenue = pg_leader * lmp_leader;
        leader_profits(k) = leader_revenue - leader_cost;

        follower_pg_results(k, :) = pg_followers';
    else
        leader_profits(k) = -inf; 
    end
end

fprintf('Simulation finished.\n\n');

%% 
[max_profit, best_idx] = max(leader_profits);
optimal_pg1 = leader_pg_range(best_idx);
optimal_pg_followers = follower_pg_results(best_idx, :)';

Pg_opt = zeros(num_gens, 1);
Pg_opt(leader_gen_idx) = optimal_pg1;
Pg_opt(follower_gen_indices) = optimal_pg_followers;

pg_leader = optimal_pg1;
n_vars_f = num_followers + n;
H_f = zeros(n_vars_f);
H_f(1:num_followers, 1:num_followers) = diag(2 * a(follower_gen_indices));
f_f = zeros(n_vars_f, 1);
f_f(1:num_followers) = b(follower_gen_indices);
lb_f = -Inf(n_vars_f, 1); ub_f = Inf(n_vars_f, 1);
lb_f(1:num_followers) = pmin(follower_gen_indices);
ub_f(1:num_followers) = pmax(follower_gen_indices);
Aeq_f = zeros(n, n_vars_f);
beq_f = Pd;
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
ref_delta_idx_f = num_followers + ref_bus;
Aeq_f = [Aeq_f; zeros(1, n_vars_f)];
Aeq_f(end, ref_delta_idx_f) = 1;
beq_f = [beq_f; 0];
[~, ~, ~, ~, lambda_final] = quadprog(H_f, f_f, [], [], Aeq_f, beq_f, lb_f, ub_f, [], options);
LMP_values = lambda_final.eqlin(1:n); 

total_gen = sum(Pg_opt);
total_demand = sum(Pd);
cost_per_gen = a .* Pg_opt.^2 + b .* Pg_opt + c;
total_cost = sum(cost_per_gen);

profit_per_gen = zeros(num_gens, 1);
for i = 1:num_gens
    gen_bus_for_profit = gen_buses(i);
    profit_per_gen(i) = (Pg_opt(i) * LMP_values(gen_bus_for_profit)) - cost_per_gen(i);
end

fprintf('---------------------------------------------------\n');
fprintf('--- Stackelberg Equilibrium Results (IEEE 14-bus) ---\n');
fprintf('---------------------------------------------------\n');
fprintf('Optimal Generator Outputs:\n');
for i = 1:num_gens
    fprintf('  Pg%d (Bus %d) = %.2f MW\n', i, gen_buses(i), Pg_opt(i));
end
fprintf('---------------------------------------------------\n');
fprintf('Total Generation = %.2f MW\n', total_gen);
fprintf('Total Demand     = %.2f MW\n', total_demand);
fprintf('Power Balance Error = %.4f MW\n', abs(total_gen - total_demand));
fprintf('---------------------------------------------------\n');
fprintf('Locational Marginal Prices (LMPs):\n');
for i = 1:n
    fprintf('  LMP at Bus %-2d = %.2f $/MWh\n', i, LMP_values(i));
end
fprintf('---------------------------------------------------\n');
fprintf('Profits per Generator:\n');
for i = 1:num_gens
    fprintf('  Profit G%d = %.2f $\n', i, profit_per_gen(i));
end
for i = 1:num_gens
    fprintf('  Cost G%d = %.2f $\n', i, cost_per_gen(i));
end
fprintf('Total System Generation Cost = %.2f $\n', total_cost);
fprintf('---------------------------------------------------\n');


%%
figure;
plot(leader_pg_range, leader_profits, 'b-', 'LineWidth', 2);
hold on;
plot(optimal_pg1, max_profit, 'r*', 'MarkerSize', 10, 'LineWidth', 2);
title('Leader''s (Gen 1) Profit vs. Generation');
xlabel('Generation of Leader (Pg1) [MW]');
ylabel('Profit [$]');
legend('Profit Curve', 'Optimal Point (Stackelberg Equilibrium)');
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
title('Optimal Power Dispatch in Stackelberg Equilibrium');
ylabel('Power Output [MW]');
grid on;
box on;