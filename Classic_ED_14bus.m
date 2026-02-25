clc;
clear;
close all;

%% 
mpc = loadcase('case14');
n = size(mpc.bus, 1);
num_gens = size(mpc.gen, 1);
% cost
gendata = mpc.gencost(:, 5:7);
%a = mpc.gencost(:, 5);
a = [0.043;0.02;0.025;0.02;0.03];
%b = mpc.gencost(:, 6);
b = [20;20;45;35;50];
%c = mpc.gencost(:, 7);
c = [0;0;0;0;0];
gen_buses = mpc.gen(:, 1);
% pmin and pmax
%pmin = mpc.gen(:, 6);
%pmax = mpc.gen(:, 7);
pmin = [5;5;5;5;5];
pmax = [100;150;200;250;250];
Pd = mpc.bus(:, 3)*3.46;
% B matrix
Bdc = makeBdc(mpc);
B = full(Bdc);
%%
n_vars = num_gens + n;

H = zeros(n_vars);
H(1:num_gens, 1:num_gens) = diag(2 * a);
f = zeros(n_vars, 1);
f(1:num_gens) = b;

lb = -Inf(n_vars, 1);
ub = Inf(n_vars, 1);

lb(1:num_gens) = pmin;
ub(1:num_gens) = pmax;

Aeq = zeros(n + 1, n_vars);
beq = zeros(n + 1, 1);

Aeq(1, 1:num_gens) = 1;
beq(1) = sum(Pd);

current_row = 2;
for i = 1:n
    gen_idx_at_bus_i = find(gen_buses == i);
    if ~isempty(gen_idx_at_bus_i)
        Aeq(current_row, gen_idx_at_bus_i) = 1;
    end
    
    Aeq(current_row, num_gens + i) = -sum(B(i, :));
    for j_bus = 1:n
        if i ~= j_bus
            Aeq(current_row, num_gens + j_bus) = Aeq(current_row, num_gens + j_bus) + B(i, j_bus);
        end
    end
    
    Aeq(current_row, num_gens + i) = Aeq(current_row, num_gens + i) - Pd(i); % ????? ???
    beq(current_row) = 0;
    
    current_row = current_row + 1;
end

ref_bus = 1; 
ref_delta_idx = num_gens + ref_bus;
Aeq = [Aeq; zeros(1, n_vars)];
Aeq(end, ref_delta_idx) = 1;
beq = [beq; 0];


%% qp optimization
options = optimoptions('quadprog', 'Display', 'off');
[x_opt, fval, exitflag, output, lambda] = quadprog(H, f, [], [], Aeq, beq, lb, ub, [], options);

if exitflag > 0
    Pg_opt = x_opt(1:num_gens);
    Delta_opt = x_opt(num_gens + 1:end);
    LMP_values = lambda.eqlin(1:n);
    
    cost_per_gen = zeros(num_gens, 1);
    for i = 1:num_gens
        cost_per_gen(i) = a(i)*Pg_opt(i)^2 + b(i)*Pg_opt(i) + c(i);
    end

    profit_per_gen = zeros(num_gens, 1);
    for i = 1:num_gens
        gen_bus_idx = gen_buses(i);
        profit_per_gen(i) = (Pg_opt(i) * LMP_values(gen_bus_idx)) - cost_per_gen(i);
    end
    
    total_generation = sum(Pg_opt);
    total_demand = sum(Pd);
    total_system_cost = sum(cost_per_gen);
    total_system_profit = sum(profit_per_gen);

    %% Results
    fprintf('---------------------------------------------------\n');
    fprintf('--- Economic Dispatch Results for IEEE 14-bus ---\n');
    fprintf('---------------------------------------------------\n');
    fprintf('Optimal Generator Outputs:\n');
    for i = 1:num_gens
        fprintf('  Pg%d (Bus %d) = %.2f MW\n', i, gen_buses(i), Pg_opt(i));
    end
    fprintf('---------------------------------------------------\n');
    fprintf('Total Generation = %.2f MW\n', total_generation);
    fprintf('Total Demand     = %.2f MW\n', total_demand);
    fprintf('Power Balance Error = %.4f MW\n', abs(total_generation - total_demand));
    fprintf('---------------------------------------------------\n');
    fprintf('Costs per Generator:\n');
    for i = 1:num_gens
        fprintf('  Cost G%d = %.2f $\n', i, cost_per_gen(i));
    end
    fprintf('Total System Generation Cost = %.2f $\n', total_system_cost);
    fprintf('---------------------------------------------------\n');
    fprintf('Locational Marginal Prices (LMPs):\n');
    for i = 1:n
        fprintf('  LMP at Bus %d = %.2f $/MWh\n', i, LMP_values(i));
    end
    fprintf('---------------------------------------------------\n');
    fprintf('Profits per Generator:\n');
    for i = 1:num_gens
        fprintf('  Profit G%d = %.2f $\n', i, profit_per_gen(i));
    end
    fprintf('Total System Profit = %.2f $\n', total_system_profit);
    fprintf('---------------------------------------------------\n');

    %% Plot LMPs
  %  figure;
   % bar(1:n, LMP_values, 'FaceColor', [0.1 0.4 0.8]);
    %xlabel('Bus Number');
    %ylabel('Locational Marginal Price (LMP) [$/MWh]');
    %title('Locational Marginal Prices (LMPs) at Each Bus (IEEE 14-bus)');
    %grid on;
    %box on;
    %xticks(1:n);
    %xticklabs = arrayfun(@(x) sprintf('Bus %d', x), 1:n, 'UniformOutput', false);
    %xticklabels(xticklabs);

else
    fprintf('\nOptimization failed. Exitflag: %d\n', exitflag);
    disp(output);
end