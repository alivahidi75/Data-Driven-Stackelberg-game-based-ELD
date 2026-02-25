%% baseline_B_SENSITIVITY.m
clc
clear
close all
%% --- 1. Load case (IEEE-14) & Settings ---
mpc = loadcase('case14');
n = size(mpc.bus, 1);
num_gens = size(mpc.gen, 1);
gen_buses = mpc.gen(:, 1);

% Cost coefficients (a*Pg^2 + b*Pg + c)
a = [0.043; 0.02; 0.025; 0.02; 0.03];
b = [20; 20; 45; 35; 50];
c = [0; 0; 0; 0; 0];

% Generation limits
pmin = [5; 5; 5; 5; 5];
pmax = [100; 150; 200; 250; 250];

Pd_base = mpc.bus(:, 3);
Pd_fixed = Pd_base;

% DC Power Flow Matrix
B = makeBdc(mpc);
B = full(B);

%% --- Leader / Follower Setup ---
leader_gen_idx = 2; % Generator 2 is the Leader
follower_gen_indices = [1; 3; 4; 5]; % Others are Followers
num_followers = length(follower_gen_indices);
leader_bus = gen_buses(leader_gen_idx);

% Uncertainty setup (20% range for follower cost coefficients)
cost_uncertainty_factor = 0.2;
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

%% --- 2. Simulation Parameters ---
n_steps = 100;
leader_pg_range = linspace(pmin(leader_gen_idx), pmax(leader_gen_idx), n_steps);
rng(0);

% M values for sensitivity analysis (Extended for better convergence)
M_range = [1000, 3000, 6000, 7000, 8000, 8500, 9000, 9500, 10000];
M_results = nan(length(M_range), 1);
M_best_PgL = nan(length(M_range), 1);
M_best_PgF = nan(length(M_range), num_followers);

fprintf('Running Sensitivity Analysis on M (Monte Carlo Scenarios)...\n');

%% --- 3. Sensitivity Analysis Loop (Outer Loop) ---
for m_idx = 1:length(M_range)
    M = M_range(m_idx);
    worst_profit = -inf(n_steps, 1);
    current_global_best_worst_profit = -inf;
    current_best_PgL = 0;
    current_best_PgF = zeros(num_followers, 1);
    
    fprintf('  Starting run for M = %d...\n', M);

    %% --- Main Loop (Inner Loop: PgL steps) ---
    for k = 1:n_steps
        PgL = leader_pg_range(k);
        profits_s = nan(M, 1);
        
        for s = 1:M % Monte Carlo Scenarios Loop
            % 1. Randomize follower costs
            a_rand = a; b_rand = b; c_rand = c;
            for i = 1:num_followers
                idx = follower_gen_indices(i);
                a_rng = follower_cost_ranges.(['a' num2str(i)]);
                b_rng = follower_cost_ranges.(['b' num2str(i)]);
                c_rng = follower_cost_ranges.(['c' num2str(i)]);
                % Uniform random sampling
                a_rand(idx) = a_rng(1) + (a_rng(2)-a_rng(1))*rand();
                b_rand(idx) = b_rng(1) + (b_rng(2)-b_rng(1))*rand();
                c_rand(idx) = c_rng(1) + (c_rng(2)-c_rng(1))*rand();
            end
            
            % 2. Follower Optimization (QP: Min Cost)
            n_vars_f = num_followers + n; % PgF variables + Theta (angles) variables
            Hf = zeros(n_vars_f);
            Hf(1:num_followers, 1:num_followers) = diag(2 * a_rand(follower_gen_indices));
            ff = zeros(n_vars_f, 1);
            ff(1:num_followers) = b_rand(follower_gen_indices);
            
            lb_f = -inf(n_vars_f, 1); ub_f = inf(n_vars_f, 1);
            lb_f(1:num_followers) = pmin(follower_gen_indices);
            ub_f(1:num_followers) = pmax(follower_gen_indices);
            
            % Equality constraints (Power Balance: Pg + Pload - B*Theta = 0)
            Aeq = zeros(n, n_vars_f);
            beq = Pd_fixed; 
            beq(leader_bus) = beq(leader_bus) - PgL; % PgL is known and moved to RHS
            
            for i = 1:n
                for j = 1:num_followers
                    if i == gen_buses(follower_gen_indices(j))
                        Aeq(i,j) = 1; % Generator j contributes to power balance at its bus
                    end
                end
                Aeq(i, num_followers+1:end) = -B(i,:); % -B*Theta term
            end
            
            ref_bus = 1; % Fixed reference bus 1 angle to 0
            Aeq = [Aeq; zeros(1,n_vars_f)];
            Aeq(end, num_followers + ref_bus) = 1;
            beq = [beq; 0];
            
            options = optimoptions('quadprog','Display','off');
            [x_f, ~, exitflag, ~, lambda] = quadprog(Hf, ff, [], [], Aeq, beq, lb_f, ub_f, [], options);
            
            if exitflag > 0
                pg_f = x_f(1:num_followers);
                lmp_curr = lambda.eqlin(leader_bus); % LMP at leader bus (dual variable)
                
                % Calculate Leader's Profit with randomized cost
                leader_cost = a_rand(leader_gen_idx)*PgL^2 + b_rand(leader_gen_idx)*PgL + c_rand(leader_gen_idx);
                leader_rev = PgL * lmp_curr;
                profits_s(s) = leader_rev - leader_cost;
            else
                profits_s(s) = -inf; % Mark failed scenario
            end
        end % End M loop (scenarios)
        
        % 3. Find worst case for this PgL (inner minimization)
        min_profit_step = min(profits_s);
        
        % 4. Check if this is the BEST of the WORST so far (Maximin)
        if min_profit_step > current_global_best_worst_profit
            current_global_best_worst_profit = min_profit_step;
            current_best_PgL = PgL;
            % Note: PgF for the Maximin point is calculated later using nominal costs
        end
    end % End n_steps loop (PgL discretization)
    
    % Store the Maximin result for the current M
    M_results(m_idx) = current_global_best_worst_profit;
    M_best_PgL(m_idx) = current_best_PgL;
    
    fprintf('  Run for M = %d finished. Maximin Profit = %.3f $\n', M, M_results(m_idx));
end

%% --- 4. Plot Convergence Results ---
figure;
semilogx(M_range, M_results, 'r-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
grid on;
box on;
xlabel('Number of Monte Carlo Scenarios (M)');
ylabel('Maximum Guaranteed Profit (Maximin) [$]');
title('Convergence of Optimal Maximin Profit vs. M (Scenario-based Robust Optimization)');
% ***???? subtitle ??? ??***
set(gca, 'Xtick', M_range);
xtickangle(45);
% 
%%
%% --- 4. Plot Convergence Results (Original Plot) ---

% ???: M_range ? M_results ???? ???? ????? ???? ??? ?????.

N_decimals = 0;
scale_factor = 10^N_decimals;
M_results_floored = floor(M_results * scale_factor) / scale_factor;


figure(4);
h = semilogx(M_range, M_results_floored, 'r-s', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
grid on;
box on;

% ??????? ???: ??????? ?? ??? ????? (Bold) ? ?????? ??????
font_size = 14; 
font_weight = 'bold';

% ??????? ???? X ? Y
xlabel('Number of Monte Carlo Scenarios (M)', 'FontWeight', font_weight, 'FontSize', font_size);
ylabel('Optimal Robust Profit (Leader) [$]', 'FontWeight', font_weight, 'FontSize', font_size);
title('Conservative Convergence Trend', 'FontWeight', font_weight, 'FontSize', font_size + 2);

% ??????? ??? ???? ?????? ? ??? ???
set(gca, ...
    'Xtick', M_range, ...
    'FontSize', font_size - 2, ...
    'FontWeight', font_weight, ...
    'LineWidth', 2); % <--- ????? ????? ??? ? ???? ???? ?? 2
xtickangle(45);

% ??????? ???? Y
ylim([-346, -342]); 

file_name = 'Conservative_Convergence_Maximin'; % ??? ???? ?????
%print(file_name, '-depsc', '-r300'); 
print(file_name, '-depsc', '-r600');

% ?? ????????? ?? saveas ??????? ????:
saveas(gcf, file_name, 'ps');
%% --- 0. ????? ???????? ???? (??? ???? ?? ???????? ????? ??? ??????? ????) ---

M_range = [1000; 3000; 6000; 7000; 8000; 8500; 9000; 9500; 10000];
M_results = [-342.649; -344.136; -344.252; -343.228; -342.837; -344.288; -344.472; -344.560; -344.967];


N_decimals = 0;
scale_factor = 10^N_decimals;
M_results_floored = floor(M_results * scale_factor) / scale_factor;


figure(4);

% ???) ????? ????? ??? (Figure Position)
set(figure(4), 'Units', 'pixels');
fig_width = 900;   % ???
fig_height = 550;  % ??????
set(figure(4), 'Position', [100, 100, fig_width, fig_height]);


% ?) ??? ??????
h = semilogx(M_range, M_results_floored, 'r-s', 'LineWidth', 3, 'MarkerSize', 6, 'MarkerFaceColor', 'r');
grid on;
box on;

% ?) ??????? ??? ? ????? ????
font_size = 12;
font_weight = 'bold';

% ??????? ???? X ? Y
xlabel('Number of Monte Carlo Scenarios (M)', 'FontWeight', font_weight, 'FontSize', font_size);
ylabel('Optimal Robust Profit (Leader) [$]', 'FontWeight', font_weight, 'FontSize', font_size);
title('Conservative Convergence Trend', 'FontWeight', font_weight, 'FontSize', font_size + 2);

% ??????? ??? ???? ?????? ? ??? ???
set(gca, ...
    'Xtick', M_range, ...
    'FontSize', font_size - 2, ...
    'FontWeight', font_weight, ...
    'LineWidth', 2); % <--- ????? ????? ??? ? ???? ???? ?? 2
xtickangle(45);

% ??????? ???? Y
ylim([-346, -342]);

file_name = 'Conservative_Convergence_Maximin'; % ??? ???? ?????

% ???) ????? ?? ???? EPS (Encapsulated PostScript) - ????? ???? ???? ??????
print(file_name, '-depsc', '-r600'); 

% ?) ????? ?? ???? PS (PostScript) - ?? ??????? ?? saveas
saveas(gcf, file_name, 'ps'); % gcf = Get Current Figure