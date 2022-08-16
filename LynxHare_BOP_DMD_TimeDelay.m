%%%%%%% LYNX-HARE DATASET %%%%%%%
%%%%  optDMD and BOP-DMD  %%%%%%%
%%%%    with TIME-DELAY   %%%%%%%

close all
clear all
clc

%% load data
addpath('./src');
X = double(load('.\data\LynxHare.mat').X);

%% load data
nt_full_train = 25;
X_train = X(:,1:nt_full_train);

h = 5;

H1 = hankel(X_train(1,1:h), X_train(1,h:end));
H2 = hankel(X_train(2,1:h), X_train(2,h:end));
H = [H1; H2];

[U,S,V] = svd(H,'econ');
V = V';

%I take 80% of the energy
threshold = 0.8;
energy_perc = diag(S)/sum(diag(S));
energy = cumsum(energy_perc);
ind_energy = numel(energy( 1:find( energy > threshold, 1 ) )) ;
ind_energy

figure(1)
plot(energy_perc,'*')
hold on
yline(energy_perc(ind_energy),'--')

% set up time dynamics

% set up time dynamics
t0 = 1845;
t_train = 1893;
T = 1903;
nt_full = size(X,2);

nt_full_train = 25;
nt = nt_full - h +1;
nt_train = nt_full_train - h +1;

ts_train_full = linspace(t0,t_train,nt_full_train);
ts_train = ts_train_full(1:end-h+1);
ts_full = linspace(t0,T,nt_full);

%% Try regular optdmd

r = ind_energy;
U = U(:,1:r);

%fit to unprojected data
imode = 2;

maxiter = 100; % maximum number of iterations
tol = 1e-3/100; % tolerance of fit
eps_stall = 1e-9; % tolerance for detecting a stalled optimization
opts = varpro_opts('maxiter',maxiter,'tol',tol,'eps_stall',eps_stall);

[w_opt,e_opt,b_opt] = optdmd(H,ts_train,r,imode,opts,[],U);

x_opt = w_opt*diag(b_opt)*exp(e_opt*ts_train);
hare_opt = [x_opt(1,:), x_opt(h,end-h+2:end)] ;
lynx_opt = [x_opt(1+h,:), x_opt(h+h,end-h+2:end)];

figure(2)
plot_list = gobjects(2, 1);
plot_list(1) = plot(ts_train,x_opt(1,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
hold on
grid on
plot_list(2) = plot(ts_train, H(1,:), 'r-','LineWidth',2);
title('Hare population - optDMD')
xlabel('year', 'FontSize',15);
xtickangle(90)
xticks(ts_train)
ylabel('# individuals [thousands]','FontSize',15);
legend(plot_list([1 2]), {'Hare real data','optDMD'} ,'Location', 'northwest')
xlim([ts_train(1),ts_train(end)])
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])

figure(3)
plot_list = gobjects(2, 1);
plot_list(1) = plot(ts_train,x_opt(2,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
hold on
grid on
plot_list(2) = plot(ts_train, H(2,:), 'r-','LineWidth',2);
title('Lynx population - optDMD')
xlabel('year', 'FontSize',15);
xtickangle(90)
xticks(ts_train)
ylabel('# individuals [thousands]','FontSize',15);
legend(plot_list([1 2]), {'Lynx real data','optDMD'} ,'Location', 'northwest')
xlim([ts_train(1),ts_train(end)])
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])


%% try bootstrapping - need to decide with or without replacement
%set seed
rng(7);

%number of time points
m = length(ts_train);
n = 2;

%number you want to choose
p = 16;
assert(m>p,'For bagging take p > m')

tot_comb =  200;

%Find all the combinations
%C = nchoosek(1:numel(ts),p);
%tot_comb = nchoosek(m,p);

% eigvalues_bag = zeros(r,tot_comb);
eigvalues_bag = [];
hare_pred = [];
lynx_pred = [];
error = [];

for j = 1:tot_comb
        %try with ioptdmd with DMD modes/evals as IC
        %select indices
        unsorted_ind = randperm(m,p);
        %unsorted_ind = C(j,:);
        %sort ind so in ascending order. NOTE: evals have variable delta t
        ind = sort(unsorted_ind);

        %create dataset for this cycle by taking aforementioned indices
        xdata_cycle = H(:,ind);
        %selected index times
        ts_ind = ts_train(ind);
        
        %I use the eigenvalues founded by optdmd on the full dataset as
        %initial guess
        [w_cycle,e1_cycle,b_cycle] = optdmd(xdata_cycle,ts_ind,r,imode,opts,[],U)%,e_opt,U);
        %[w_cycle,e1_cycle,b_cycle] = optdmd(xdata_cycle,ts_ind,r,imode,[],e_opt); %varpro_opts('ifprint',0),e_opt)

        x1 = w_cycle*diag(b_cycle)*exp(e1_cycle*ts_train_full);
        
        hare_estimate = [x1(1,:), x1(h,end-h+1:end)] ;
        lynx_estimate = [x1(1+h,:), x1(h+h,end-h+1:end)];
        X_estimate = [hare_estimate; lynx_estimate];
        relerr_r = norm(X_estimate-X,'fro')/norm(X,'fro')
        error = [error relerr_r];
        if relerr_r < 0.5
            hare_pred = [hare_pred; hare_estimate];
            lynx_pred = [lynx_pred; lynx_estimate];
            eigvalues_bag = [eigvalues_bag; e1_cycle];
        end
        
end

x_opt = w_opt*diag(b_opt)*exp(e_opt*ts_train_full);
hare_opt = [x_opt(1,:), x_opt(h,end-h+1:end)] ;
lynx_opt = [x_opt(1+h,:), x_opt(h+h,end-h+1:end)];

figure(4)
% for i = 1:tot_comb
%     plot(ts_full, hare_pred(i,:))
% end
hare_mean = mean(hare_pred);
hare_std = 2 * std(hare_pred);
x2 = [ts_full, fliplr(ts_full)];
plot_list = gobjects(4, 1);
inBetween = [hare_mean - hare_std, fliplr(hare_mean + hare_std)];
plot_list(4) = fill(x2, inBetween, [0.95, 0.8, 0.95],'EdgeColor','none');
hold on
xline(1893, '--k', 'Train', 'LabelHorizontalAlignment','left', 'LabelOrientation','horizontal', 'FontSize',14);
xline(1893, '--k', 'Test', 'LabelHorizontalAlignment','right','LabelOrientation','horizontal', 'FontSize',14);
plot_list(2) = plot(ts_full, hare_mean,'k-','LineWidth',2);
plot_list(3) = plot(ts_full, hare_opt, 'r-','LineWidth',2);
plot_list(1) = plot(ts_full, X(1,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
% plot_list(2) = plot(ts_train_full, hare_mean(1:nt_full_train),'k-','LineWidth',1.4);
% plot_list(3) = plot(ts_train_full, hare_opt(1:nt_full_train), 'r-','LineWidth',1.4);
% plot(ts_full(nt_full_train:end), hare_mean(nt_full_train:end),'k--.','LineWidth',1.4)
% plot(ts_full(nt_full_train:end), hare_opt(nt_full_train:end), 'r--.','LineWidth',1.4)
grid on
xlabel('year', 'FontSize',15); 
xtickangle(90)
ylim([-85,180])
xticks(ts_full)
xlim([ts_full(1),ts_full(end)])
ylabel('# individuals [thousands]','FontSize',15);  
legend(plot_list([1 3 2 4]), {'Hare real data','optDMD','mean bopDMD', '>75% trajectories'} ,'Location', 'southwest')
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])


figure(5)
% for i = 1:tot_comb
%     plot(ts_full, lynx_pred(i,:))
% end
lynx_mean = mean(lynx_pred);
lynx_std = 2 * std(lynx_pred);
x2 = [ts_full, fliplr(ts_full)];
plot_list = gobjects(4, 1);
inBetween = [lynx_mean - lynx_std, fliplr(lynx_mean + lynx_std)];
plot_list(4) = fill(x2, inBetween, [0.95, 0.8, 0.95],'EdgeColor','none');
hold on
xline(1893, '--k', 'Train', 'LabelHorizontalAlignment','left', 'LabelOrientation','horizontal', 'FontSize',14);
xline(1893, '--k', 'Test', 'LabelHorizontalAlignment','right','LabelOrientation','horizontal', 'FontSize',14);
plot_list(2) = plot(ts_full, lynx_mean,'k-','LineWidth',2);
plot_list(3) = plot(ts_full, lynx_opt, 'r-','LineWidth',2);
plot_list(1) = plot(ts_full, X(2,:),'--.', 'LineWidth',0.1, 'Color','Blue', 'MarkerSize', 20);
% plot_list(2) = plot(ts_train_full, lynx_mean(1:nt_full_train),'k-','LineWidth',1.4);
% plot_list(3) = plot(ts_train_full, lynx_opt(1:nt_full_train), 'r-','LineWidth',1.4);
% plot(ts_full(nt_full_train:end), lynx_mean(nt_full_train:end),'k--.','LineWidth',1.4)
% plot(ts_full(nt_full_train:end), lynx_opt(nt_full_train:end), 'r--.','LineWidth',1.4)
grid on
xlabel('year', 'FontSize',15); 
xtickangle(90)
xticks(ts_full)
xlim([ts_full(1),ts_full(end)])
ylim([-10,95])
ylabel('# individuals [thousands]','FontSize',15);  
legend(plot_list([1 3 2 4]), {'Lynx real data','optDMD','mean bopDMD', '>75% trajectories'} ,'Location', 'northwest')
ax = gca;
ax.FontSize = 12; 
set(gcf,'Position',[200, 200,  650, 380])