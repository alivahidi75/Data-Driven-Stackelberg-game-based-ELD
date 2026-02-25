clc;
close all;
clear;

%% 
load('stackelberg_optimal_points_dataset.mat'); 

%% 
[X_norm, ps_x] = mapminmax(X_nn_input', 0, 1); 
[Y_norm, ps_y] = mapminmax(Y_nn_outputs', 0, 1); 

%% 
n_samples = size(X_nn_input, 1); 
idx = randperm(n_samples); 

train_ratio = 0.8;
n_train = round(train_ratio * n_samples);

train_idx = idx(1:n_train);
test_idx = idx(n_train+1:end);

X_train = X_norm(:, train_idx);
Y_train = Y_norm(:, train_idx);
X_test = X_norm(:, test_idx);
Y_test = Y_norm(:, test_idx);

%% 
net = fitnet([60 40], 'trainlm'); 

net.trainParam.epochs = 500;
net.trainParam.showWindow = true; 
net.performFcn = 'mse'; 
net.divideParam.trainRatio = 0.8; 
net.divideParam.valRatio = 0.1; 
net.divideParam.testRatio = 0.1; 

[net, tr] = train(net, X_train, Y_train);

save('trainedNet_14bus_optimized.mat', 'net', 'ps_x', 'ps_y'); 

%% 
Y_pred_norm = net(X_test);
Y_pred = mapminmax('reverse', Y_pred_norm, ps_y);
Y_test_real = mapminmax('reverse', Y_test, ps_y);

%% 
LMP1_true = round(Y_test_real(1,:)',4);
LMP1_pred = round(Y_pred(1,:)',4);

P1_true = round(Y_test_real(2,:)',4);
P1_pred = round(Y_pred(2,:)',4);

P3_true = round(Y_test_real(3,:)',5);
P3_pred = round(Y_pred(3,:)',5);

P4_true = round(Y_test_real(4,:)',5);
P4_pred = round(Y_pred(4,:)',5);

P5_true = round(Y_test_real(5,:)',4);
P5_pred = round(Y_pred(5,:)',4);

%%
fprintf('--- Neural Network Evaluation (14-Bus System) ---\n');
MAE_p1 = mean(abs(P1_true - P1_pred));
MAE_p3 = mean(abs(P3_true - P3_pred));
MAE_p4 = mean(abs(P4_true - P4_pred));
MAE_p5 = mean(abs(P5_true - P5_pred));
MAE_lmp1 = mean(abs(LMP1_true - LMP1_pred));
fprintf('MAE: LMP1=%.4f, P1=%.4f, P3=%.4f, P4=%.4f, P5=%.4f\n', MAE_lmp1, MAE_p1, MAE_p3, MAE_p4, MAE_p5);

RMSE_p1 = sqrt(mean((P1_true - P1_pred).^2));
RMSE_p3 = sqrt(mean((P3_true - P3_pred).^2));
RMSE_p4 = sqrt(mean((P4_true - P4_pred).^2));
RMSE_p5 = sqrt(mean((P5_true - P5_pred).^2));
RMSE_lmp1 = sqrt(mean((LMP1_true - LMP1_pred).^2));
fprintf('RMSE: LMP1=%.4f, P1=%.4f, P3=%.4f, P4=%.4f, P5=%.4f\n', RMSE_lmp1, RMSE_p1, RMSE_p3, RMSE_p4, RMSE_p5);

R2_p1 = 1 - sum((P1_true - P1_pred).^2) / sum((P1_true - mean(P1_true)).^2);
R2_p3 = 1 - sum((P3_true - P3_pred).^2) / sum((P3_true - mean(P3_true)).^2);
R2_p4 = 1 - sum((P4_true - P4_pred).^2) / sum((P4_true - mean(P4_true)).^2);
R2_p5 = 1 - sum((P5_true - P5_pred).^2) / sum((P5_true - mean(P5_true)).^2);
R2_lmp1 = 1 - sum((LMP1_true - LMP1_pred).^2) / sum((LMP1_true - mean(LMP1_true)).^2);
fprintf('R^2: LMP1=%.4f, P2=%.4f, P3=%.4f, P4=%.4f, P5=%.4f\n', R2_lmp1, R2_p1, R2_p3, R2_p4, R2_p5);
fprintf('--------------------------------------------------\n');


%% 
figure;
sgtitle('Scatter Plots: Actual vs. Predicted Values (Test Data)');

subplot(2,3,1);
scatter(LMP1_true, LMP1_pred, 40, 'g', 'filled'); hold on;
plot([min(LMP1_true) max(LMP1_true)], [min(LMP1_true) max(LMP1_true)], 'k--', 'LineWidth', 1.5);
xlabel('Actual LMP1 ($/MWh)'); ylabel('Predicted LMP1 ($/MWh)');
%title(sprintf('LMP1: R^2 = %.4f', R2_lmp1));
grid on; axis equal;
xlim([min(LMP1_true) max(LMP1_true)]); ylim([min(LMP1_true) max(LMP1_true)]);

subplot(2,3,2);
scatter(round(P1_true,1), round(P1_pred,1), 40, 'b', 'filled'); hold on;
plot([min(P1_true) max(P1_true)], [min(P1_true) max(P1_true)], 'k--', 'LineWidth', 1.5);
xlabel('Actual P1 (MW)'); ylabel('Predicted P1 (MW)');
%title(sprintf('P1: R^2 = %.4f', 1-R2_p1));
grid on; axis equal;
xlim([min(P1_true) max(P1_true)]); ylim([min(P1_true) max(P1_true)]);

subplot(2,3,3);
scatter(P3_true, P3_pred, 40, 'r', 'filled'); hold on;
plot([min(P3_true) max(P3_true)], [min(P3_true) max(P3_true)], 'k--', 'LineWidth', 1.5);
xlabel('Actual P3 (MW)'); ylabel('Predicted P3 (MW)');
%title(sprintf('P3: R^2 = %.4f', R2_p3));
grid on; axis equal;
xlim([min(P3_true) max(P3_true)]); ylim([min(P3_true) max(P3_true)]);

subplot(2,3,4);
scatter(P4_true, P4_pred, 40, 'm', 'filled'); hold on;
plot([min(P4_true) max(P4_true)], [min(P4_true) max(P4_true)], 'k--', 'LineWidth', 1.5);
xlabel('Actual P4 (MW)'); ylabel('Predicted P4 (MW)');
%title(sprintf('P4: R^2 = %.4f', R2_p4));
grid on; axis equal;
xlim([min(P4_true) max(P4_true)]); ylim([min(P4_true) max(P4_true)]);

subplot(2,3,5);
scatter(P5_true, P5_pred, 40, 'c', 'filled'); hold on;
plot([min(P5_true) max(P5_true)], [min(P5_true) max(P5_true)], 'k--', 'LineWidth', 1.5);
xlabel('Actual P5 (MW)'); ylabel('Predicted P5 (MW)');
%title(sprintf('P5: R^2 = %.4f', R2_p5));
grid on; axis equal;
xlim([min(P5_true) max(P5_true)]); ylim([min(P5_true) max(P5_true)]);

%% 
figure;
sgtitle('Actual vs. Predicted Values (Test Data)');

subplot(3,2,1);
plot(LMP1_true, 'b-o', 'DisplayName', 'Actual LMP1'); hold on;
plot(LMP1_pred, 'g-*', 'DisplayName', 'Predicted LMP1');
legend('Location', 'best');
xlabel('Sample Index'); ylabel('LMP ($/MWh)');
%title(sprintf('LMP1 (R^2 = %.4f)', R2_lmp1));
grid on;

subplot(3,2,2);
plot(round(P1_true,8), 'b-o', 'DisplayName', 'Actual P1'); hold on;
plot(round(P1_pred,8), 'r-*', 'DisplayName', 'Predicted P1');
legend('Location', 'best');
xlabel('Sample Index'); ylabel('Power (MW)');
%title(sprintf('P1 (R^2 = %.4f)', R2_p1));
grid on;

subplot(3,2,3);
plot(P3_true, 'b-o', 'DisplayName', 'Actual P3'); hold on;
plot(P3_pred, 'r-*', 'DisplayName', 'Predicted P3');
legend('Location', 'best');
xlabel('Sample Index'); ylabel('Power (MW)');
%title(sprintf('P3 (R^2 = %.4f)', R2_p3));
grid on;

subplot(3,2,4);
plot(P4_true, 'b-o', 'DisplayName', 'Actual P4'); hold on;
plot(P4_pred, 'm-*', 'DisplayName', 'Predicted P4');
legend('Location', 'best');
xlabel('Sample Index'); ylabel('Power (MW)');
%title(sprintf('P4 (R^2 = %.4f)', R2_p4));
grid on;

subplot(3,2,5);
plot(P5_true, 'b-o', 'DisplayName', 'Actual P5'); hold on;
plot(P5_pred, 'c-*', 'DisplayName', 'Predicted P5');
legend('Location', 'best');
xlabel('Sample Index'); ylabel('Power (MW)');
%title(sprintf('P5 (R^2 = %.4f)', R2_p5));
grid on;