function [FA, NE, TT] = Sweep_Files(filename1, config)

disp(filename1);

obj0 = sparameters(filename1);
s_freq = obj0.Frequencies;
s_freq = s_freq';
portOrder = config(6);
sweepFileFigure = config(7);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Sweep all the victims: Port order 1 or 2 %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Get the port number %%%
s0 = obj0.Parameters;
portNum = min(size(s0));
% diff_pt_num = portNum*0.5;

if(portOrder==2) % 1->N THU 
    victimSEQ = [1:1:(0.5*portNum)];
end

if(portOrder==1) %  1->2 THU 
    victimSEQ = [1:2:(portNum)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Get ICN for all the victims %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for v = 1:length(victimSEQ)
    [ct_nx(v, :), ct_fx(v, :), ct_total(v, :)] = ICN(filename1, victimSEQ(v), config);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  OUTPUT: TXT files &. Figures  %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sampleNum = min(size(ct_nx));
%%%%%%% Final Value %%%%%%%%%%%
for d = 1: sampleNum
    final_ct_nx(d) = ct_nx(d, end);
    final_ct_fx(d) = ct_fx(d, end);
    final_ct_total(d) = ct_total(d, end);
end

%%%%%%%%%%% TXT output %%%%%%%%%
NUM = [1:1:sampleNum];
FA = final_ct_fx;
NE = final_ct_nx;
TT = final_ct_total;
TXTname = filename1(1:(end-5));
% txtOutput(NUM, FA, NE, TT, TXTname);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Plot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(sweepFileFigure == 1)
    color = ['b','k','r','b*',':k', 'g',':g', 'c', 'y', 'm',':r'];
    figure()
    for a = 1: sampleNum
      plot(s_freq, ct_nx(a, :), color(a), 'LineWidth', 3.5)
      hold on; 
    end
    grid on;
    titleNEXT = ['Near-end ICN: ', TXTname];
    title(titleNEXT,'fontsize',13);
    xlabel('Freqeuncy [Hz]','fontsize',13);
    ylabel('Voltage [V]','fontsize',13);

    figure()
    for b = 1: sampleNum
      plot(s_freq, ct_fx(b, :), color(b), 'LineWidth', 3.5)
      hold on; grid on;
    end
    titleFEXT = ['Far-end ICN: ', TXTname];
    title(titleFEXT,'fontsize',13);
    xlabel('Freqeuncy [Hz]','fontsize',13);
    ylabel('Voltage [V]','fontsize',13);

    figure()
    for c = 1: sampleNum
      plot(s_freq, ct_total(c, :), color(c), 'LineWidth', 3.5)
      hold on; grid on;
    end
    titleTT = ['Total ICN: ', TXTname];
    title(titleTT,'fontsize',13);
    xlabel('Freqeuncy [Hz]','fontsize',13);
    ylabel('Voltage [V]','fontsize',13);

    %%%%%%%%%%%% Plot BARS %%%%%%%%%%%%%
    barWidth = 0.3;
    %%% Near-end %%%
    figure()
    bar([1:1:sampleNum], final_ct_nx(1:sampleNum), barWidth, 'b');
    title(titleNEXT,'fontsize',13);
    xlabel('Layout','fontsize',13);
    ylabel('Voltage[Volts]','fontsize',13);
    grid on;
    ylim([0,1.05*max(final_ct_nx)]);

    %%% Far-end %%%
    figure()
    bar([1:1:sampleNum], final_ct_fx(1:sampleNum), barWidth, 'b');
    title(titleFEXT,'fontsize',13);
    xlabel('Layout','fontsize',13);
    ylabel('Voltage[Volts]','fontsize',13);
    grid on;
    ylim([0,1.05*max(final_ct_fx)]);

    %%% Total %%%
    figure()
    bar([1:1:sampleNum], final_ct_total(1:sampleNum), barWidth, 'b');
    title(titleTT,'fontsize',13);
    xlabel('Layout','fontsize',13);
    ylabel('Voltage[Volts]','fontsize',13);
    grid on;
    ylim([0,1.05*max(final_ct_total)]);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

end

