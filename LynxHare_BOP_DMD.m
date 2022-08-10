%%%%%%% LYNX-HARE DATASET %%%%%%%
%%%%  optDMD and BOP-DMD  %%%%%%%

close all
clear all
clc

%% load data
addpath('./src');
X = double(load('.\data\LynxHare.mat').X);

% set up time dynamics
t0 = 1845;
T = 1903;
nt = size(X,2);
ts = linspace(t0,T,nt);

% create train set
nt_train = 25;
ts_train = ts(1:nt_train);
X_train = X(:,1:nt_train);

%% Try regular optdmd
n = 2;  %# states
r = 2;  %latent dimension
imode = 1;

[w_opt,e_opt,b_opt] = optdmd(X_train,ts_train,r,imode);

% reconstructed values
x_opt = w_opt*diag(b_opt)*exp(e_opt*ts);
relerr_r = norm(x_opt-X,'fro')/norm(X,'fro');

% plot
figure(1)
plot_list = gobjects(2, 1);
plot_list(1) = plot(ts, X(1,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
hold on
grid on
plot_list(2) = plot(ts, x_opt(1,:), 'r-','LineWidth',2);
xline(1893, '--k', 'Train', 'LabelHorizontalAlignment','left', 'LabelOrientation','horizontal', 'FontSize',14);
xline(1893, '--k', 'Test', 'LabelHorizontalAlignment','right','LabelOrientation','horizontal', 'FontSize',14);;
title('Hare population - optDMD')
xlabel('year', 'FontSize',15);
xtickangle(90)
xticks(ts)
ylabel('# individuals [thousands]','FontSize',15);
legend(plot_list([1 2]), {'Hare real data','optDMD'} ,'Location', 'northwest')
xlim([ts(1),ts(end)])
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])

figure(2)
plot_list = gobjects(2, 1);
plot_list(1) = plot(ts, X(2,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
hold on
grid on
plot_list(2) = plot(ts, x_opt(2,:), 'r-','LineWidth',2);
xline(1893, '--k', 'Train', 'LabelHorizontalAlignment','left', 'LabelOrientation','horizontal', 'FontSize',14);
xline(1893, '--k', 'Test', 'LabelHorizontalAlignment','right','LabelOrientation','horizontal', 'FontSize',14);;
title('Lynx population - optDMD')
xlabel('year', 'FontSize',15);
xtickangle(90)
xticks(ts)
ylabel('# individuals [thousands]','FontSize',15);
legend(plot_list([1 2]), {'Lynx real data','optDMD'} ,'Location', 'northwest')
xlim([ts(1),ts(end)])
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])


%% try bootstrapping without replacement

%set seed
rng(7);

p = 21; %bootstrap sub-sample size
assert(p<nt_train, 'Sub-sample size larger than sample size')

tot_comb =  100;

eigvalues_bag = zeros(r,tot_comb);
%eigvector_bag = zeros(r,tot_comb);
hare_pred = zeros(tot_comb, nt);
lynx_pred = zeros(tot_comb, nt);

for j = 1:tot_comb
        %select indices
        unsorted_ind = randperm(nt_train,p);

        %sort ind so in ascending order.
        ind = sort(unsorted_ind);

        %create dataset for this cycle by taking aforementioned indices
        xdata_cycle = X_train(:,ind);

        %selected index times
        ts_ind = ts_train(ind);
        
        [w_cycle,e1_cycle,b_cycle] = optdmd(xdata_cycle,ts_ind,r,imode);

        %save eigenvalues
        eigvalues_bag(:,j) = e1_cycle;

        %compute and save trajectories
        x1 = w_cycle*diag(b_cycle)*exp(e1_cycle*ts);

        hare_pred(j,:) = x1(1,:);
        lynx_pred(j,:) = x1(2,:);
end

%plot
figure(3)
hare_mean = mean(hare_pred);
hare_std = 2 * std(hare_pred);
hare_opt = x_opt(1,:);
x2 = [ts, fliplr(ts)];
plot_list = gobjects(4, 1);
inBetween = [hare_mean - hare_std, fliplr(hare_mean + hare_std)];
plot_list(4) = fill(x2, inBetween, [0.95, 0.8, 0.95],'EdgeColor','none');
hold on
xline(1893, '--k', 'Train', 'LabelHorizontalAlignment','left', 'LabelOrientation','horizontal', 'FontSize',14);
xline(1893, '--k', 'Test', 'LabelHorizontalAlignment','right','LabelOrientation','horizontal', 'FontSize',14);
plot_list(1) = plot(ts, X(1,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
plot_list(2) = plot(ts, hare_mean,'k-','LineWidth',2);
plot_list(3) = plot(ts, hare_opt, 'r-','LineWidth',2);
grid on
xlabel('year', 'FontSize',15);
xtickangle(90)
xticks(ts)
ylabel('# individuals [thousands]','FontSize',15);
title('Hare population - BOP-DMD')
legend(plot_list([1 3 2 4]), {'Hare real data','optDMD','mean bopDMD', '>75% trajectories'} ,'Location', 'southwest')
ylim([-85,180])
xlim([ts(1),ts(end)])
box on
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])
    
    
figure(4)
lynx_mean = mean(lynx_pred);
lynx_std = 2 * std(lynx_pred);
lynx_opt = x_opt(2,:);
x2 = [ts, fliplr(ts)];
plot_list = gobjects(4, 1);
inBetween = [lynx_mean - lynx_std, fliplr(lynx_mean + lynx_std)];
plot_list(4) = fill(x2, inBetween, [0.95, 0.8, 0.95],'EdgeColor','none');
hold on
xline(1893, '--k', 'Train', 'LabelHorizontalAlignment','left', 'LabelOrientation','horizontal', 'FontSize',14);
xline(1893, '--k', 'Test', 'LabelHorizontalAlignment','right','LabelOrientation','horizontal', 'FontSize',14);
plot_list(1) = plot(ts, X(2,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
plot_list(2) = plot(ts, lynx_mean,'k-','LineWidth',2);
plot_list(3) = plot(ts, lynx_opt, 'r-','LineWidth',2);
grid on
xlabel('year', 'FontSize',15);
title('Lynx population - BOP-DMD')
xtickangle(90)
xticks(ts)
ylabel('# individuals [thousands]','FontSize',15);
legend(plot_list([1 3 2 4]), {'Lynx real data','optDMD','mean bopDMD', '>75% trajectories'} ,'Location', 'northwest')
ylim([-10,95])
xlim([ts(1),ts(end)])
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])


