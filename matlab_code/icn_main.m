function [icn] = icn_main(filename, fb)     
    ft = fb; 
    fr = 0.75*fb; % the cut-off freq for the receiving filter [GHz]
    Ant = 1000*(10^-3); %Disturber Amplitude @near end [v]
    Aft = 1000*(10^-3);% Disturber Amplitude @far end [v]
    portOrder = 2; 
    sweepFileFigure = 0;
    mode = 1; %%
    config = [Ant, Aft, fb, ft, fr, portOrder, sweepFileFigure, mode];
    [FA, NE, TT] = Sweep_Files(char(filename), config);
    disp(FA)
    icn = mean(FA)
end