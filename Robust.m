%% Robust KKT -> MPEC Implementation (Method A) using YALMIP
% Solves: max_PgL min_xi Profit_L(PgL, PgF(xi)) s.t. KKT(PgF) holds for xi

clc;
clear;
% close all; % Commented out for debugging environment

% --- 1. Load Case (IEEE-14) & Constants ---
mpc = loadcase('case14');
n = size(mpc.bus, 1);
num_gens = size(mpc.gen, 1);
gen_buses = mpc.gen(:, 1);

% Cost Coefficients (EXACTLY AS IN YOUR CODE)
a = [0.043; 0.02; 0.025; 0.02; 0.03];
b = [20; 20; 45; 35; 50];
c = [0; 0; 0; 0; 0];

% Pmin / Pmax (EXACTLY AS IN YOUR CODE)
pmin = [5; 5; 5; 5; 5];
pmax = [100; 150; 200; 250; 250];

% Fixed Base Load
Pd_fixed = mpc.bus(:, 3); 

% B Matrix
B = makeBdc(mpc);
B = full(B);

% Find Ref Bus: Type 3 is reference bus
ref_bus_type3 = find(mpc.bus(:, 2) == 3, 1);
ref_bus = ref_bus_type3;
if isempty(ref_bus)
    ref_bus = 1; % Fallback to Bus 1
end

% --- Leader / Follower Setup ---
leader_gen_idx = 2;
follower_gen_indices = [1; 3; 4; 5];
num_followers = length(follower_gen_indices);
leader_bus = gen_buses(leader_gen_idx);

% --- BIG M Constant (MUST BE LARGE ENOUGH) ---
M_big = 1000; 

% --- Uncertainty Settings (Follower Cost Range) ---
cost_uncertainty_factor = 0.2; % +/- 20%
a_base = a(follower_gen_indices);
b_base = b(follower_gen_indices);

a_min = a_base * (1 - cost_uncertainty_factor);
a_max = a_base * (1 + cost_uncertainty_factor);
b_min = b_base * (1 - cost_uncertainty_factor);
b_max = b_base * (1 + cost_uncertainty_factor);

%% --- 2. YALMIP Variable Definition (Primal, Dual, Binary, Uncertainty) ---
% Primal Variables
PgL = sdpvar(1, 1); 
PgF = sdpvar(num_followers, 1);
theta = sdpvar(n, 1);

% Dual Variables
lambda = sdpvar(n, 1);
mu_min = sdpvar(num_followers, 1); 
mu_max = sdpvar(num_followers, 1); 

% Binary Variables (Big-M)
z_min = binvar(num_followers, 1); 
z_max = binvar(num_followers, 1); 

% Uncertainty Variables
aF_unc = sdpvar(num_followers, 1); 
bF_unc = sdpvar(num_followers, 1); 

% Maximin Profit Variable
gamma = sdpvar(1, 1); 


%% --- 3. Constraints (KKT, Feasibility, Uncertainty Set) ---
Constraints = [];

% --- A. Follower Primal Feasibility (Power Balance) ---
% We construct the constant parts (Aeq_F_const and beq_F_const) first, 
% then combine with sdpvars to avoid NaN contamination.

% Power Balance Matrix Aeq: [PgF | theta]
Aeq_F = zeros(n, num_followers + n);

for i = 1:n
    % Generator contribution (PgF)
    for j = 1:num_followers
        follower_bus = gen_buses(follower_gen_indices(j));
        if i == follower_bus
            Aeq_F(i, j) = 1;
        end
    end
    % Angle contribution (-B*theta)
    Aeq_F(i, num_followers+1:end) = -B(i, :);
end

% Combine constant Aeq_F with sdpvar: Aeq_F * [PgF; theta]
Primal_Balance_LHS = Aeq_F * [PgF; theta];

% --- CRITICAL FIX APPLIED HERE ---
% 1. Start with fixed load vector
Primal_Balance_RHS = Pd_fixed; 

% 2. Create the variable contribution vector: -PgL at leader_bus, 0 elsewhere
PgL_contribution = zeros(n, 1);
PgL_contribution(leader_bus) = PgL;

% 3. Final RHS: Load - Leader's Generation
Primal_Balance_RHS = Primal_Balance_RHS - PgL_contribution;
% --- END CRITICAL FIX ---

% Final Balance Constraint
Constraints = [Constraints, Primal_Balance_LHS == Primal_Balance_RHS];

% Ref Bus constraint (theta_ref = 0)
ref_idx_in_theta = ref_bus;
Constraints = [Constraints, theta(ref_idx_in_theta) == 0];

% Leader Pmin/Pmax 
Constraints = [Constraints, pmin(leader_gen_idx) <= PgL <= pmax(leader_gen_idx)];

% Follower Pmin/Pmax (Primal KKT feasibility)
pmin_F = pmin(follower_gen_indices);
pmax_F = pmax(follower_gen_indices);
Constraints = [Constraints, pmin_F <= PgF <= pmax_F];

% --- B. Follower Dual Feasibility ---
Constraints = [Constraints, mu_min >= 0, mu_max >= 0];


% --- C. Follower Stationarity ---
% 1. W.r.t PgF: 2*aF_unc*PgF + bF_unc - lambda_bus - mu_max + mu_min = 0
lambda_F_bus = lambda(gen_buses(follower_gen_indices));
Constraints_Stat_PgF = 2 * diag(aF_unc) * PgF + bF_unc - lambda_F_bus - mu_max + mu_min == 0;
Constraints = [Constraints, Constraints_Stat_PgF];

% 2. W.r.t theta: -B'*lambda = 0 
Constraints = [Constraints, -B' * lambda == 0];

% --- D. Follower Complementarity (using Big-M MIQP/MILP) ---
Constraints = [Constraints, mu_min <= M_big * z_min];
Constraints = [Constraints, (PgF - pmin_F) <= M_big * (1 - z_min)];
Constraints = [Constraints, mu_max <= M_big * z_max];
Constraints = [Constraints, (pmax_F - PgF) <= M_big * (1 - z_max)];

% --- E. Uncertainty Set Constraints ---
Constraints_Uncertainty = [a_min <= aF_unc <= a_max];
Constraints_Uncertainty = [Constraints_Uncertainty, b_min <= bF_unc <= b_max];


%% --- 4. Maximin Objective Formulation ---
% Leader's Profit Expression (Profit_L)
Cost_L = a(leader_gen_idx) * PgL^2 + b(leader_gen_idx) * PgL + c(leader_gen_idx);
Revenue_L = PgL * lambda(leader_bus); 
Profit_L = Revenue_L - Cost_L;

% Maximize gamma subject to gamma <= Profit_L for all uncertainty
Constraints = [Constraints, Constraints_Uncertainty]; 
Constraints = [Constraints, gamma <= Profit_L];

Objective = gamma;

%% --- 5. Solving the Robust MPEC ---

options = sdpsettings('solver', 'gurobi', 'verbose', 0, 'robust', 1); % robust=1 enables robust formulation
% If Gurobi/Cplex/Mosek are not available, you might try fmincon/bmibnb, but stability is low.

% Define the full set of constraints including KKT and uncertainty bounds
F = Constraints;

sol = solvesdp(F, -Objective, options); 

%% --- 6. Extract and Report Results ---

if sol.problem == 0
    % Optimal Maximin Solution
    Optimal_PgL = value(PgL);
    Optimal_Gamma = value(gamma);
    
    % --- Re-solve nominal QP for LMPs (for reporting only) ---
    pmin_F = pmin(follower_gen_indices);
    pmax_F = pmax(follower_gen_indices);
    
    n_vars_f = num_followers + n;
    H_f = zeros(n_vars_f);
    H_f(1:num_followers, 1:num_followers) = diag(2 * a(follower_gen_indices)); % Nominal 'a'
    f_f = zeros(n_vars_f, 1);
    f_f(1:num_followers) = b(follower_gen_indices); % Nominal 'b'
    
    lb_f = -Inf(n_vars_f, 1); ub_f = Inf(n_vars_f, 1);
    lb_f(1:num_followers) = pmin_F;
    ub_f(1:num_followers) = pmax_F;
    
    Aeq_f = Aeq_F;
    beq_f_qp = Pd_fixed;
    beq_f_qp(leader_bus) = beq_f_qp(leader_bus) - Optimal_PgL;
    
    Aeq_f = [Aeq_f; zeros(1, n_vars_f)];
    Aeq_f(end, num_followers + ref_bus) = 1;
    beq_f_qp = [beq_f_qp; 0];
    
    options_qp = optimoptions('quadprog', 'Display', 'off');
    [x_f_opt, ~, exitflag, ~, lambda_final] = quadprog(H_f, f_f, [], [], Aeq_f, beq_f_qp, lb_f, ub_f, [], options_qp);
    
    if exitflag > 0
        Optimal_PgF = x_f_opt(1:num_followers);
        LMP_values_nominal = lambda_final.eqlin(1:n);
        
        % Full generation vector (using nominal follower response)
        Pg_opt = zeros(num_gens, 1);
        Pg_opt(leader_gen_idx) = Optimal_PgL;
        Pg_opt(follower_gen_indices) = Optimal_PgF;
        
        % Costs and Profits (using NOMINAL costs for general report)
        cost_per_gen = a .* Pg_opt.^2 + b .* Pg_opt + c;
        total_cost = sum(cost_per_gen);
        
        profit_per_gen = zeros(num_gens, 1);
        for i = 1:num_gens
            gen_bus_for_profit = gen_buses(i);
            profit_per_gen(i) = (Pg_opt(i) * LMP_values_nominal(gen_bus_for_profit)) - cost_per_gen(i);
        end
        
        
        fprintf('---------------------------------------------------\n');
        fprintf('--- Robust Stackelberg MPEC (Method A) Results ---\n');
        fprintf('---------------------------------------------------\n');
        fprintf('Maximin Leader Strategy (Guaranteed Profit):\n');
        fprintf('  Optimal Pg%d (Bus %d) = %.3f MW\n', leader_gen_idx, leader_bus, Optimal_PgL);
        fprintf('  Guaranteed Worst-Case Profit (Gamma) = %.3f $\n', Optimal_Gamma);
        fprintf('---------------------------------------------------\n');
        
        fprintf('Note: Following values use the calculated Optimal PgL and\n');
        fprintf('