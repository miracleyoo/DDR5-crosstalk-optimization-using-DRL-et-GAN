function [ct_nx, ct_fx, ct_total ] = ICN(filename, victim, config)


obj = sparameters(filename);
f = obj.Frequencies;
s0 = obj.Parameters;

Ant = config(1); %Disturber Amplitude @near end [mv]
Aft = config(2);% Disturber Amplitude @far end [mv]
fb = config(3); % data rate 30[G bps]
ft = config(4); 
fr = config(5); % the cut-off freq for the receiving filter [GHz]
ptOrder = config(6);
mode = config(8);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%   Select Modes   %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Single Ended Mode: mode=1  %%% 
if(mode == 1)
    sdd = s0;
end
%%% Differential Mode: mode=2  %%% 
if(mode == 2)
    [sdd, sdc, scd, scc] = s2smm(s0,ptOrder); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Weight function   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1: length(f)
   weightNT(i) = ((Ant^2)/fb)* ((sinc(f(i)/fb))^2)* ( 1/(1 + (f(i)/ft)^4))* ( 1/(1+(f(i)/fr)^8) );
   weightFT(i)  = ((Aft^2)/fb)* ((sinc(f(i)/fb))^2)* ( 1/(1 + (f(i)/ft)^4))* ( 1/(1+(f(i)/fr)^8) );
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Multi-disturber crosstalk  loss %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[MDNEXT, MDFEXT ] = MDXT(sdd, victim, ptOrder);
deltaF = f(2)-f(1);
sum_nx = 0;% Initial Value of sum
sum_fx = 0;
for j = 1: length(f)
   sum_nx = sum_nx + weightNT(j)*10^(-0.1*MDNEXT(j)); 
   ct_nx(j) = (2*deltaF * sum_nx).^(0.5); % Near-end Crosstalk
   
   sum_fx = sum_fx + weightFT(j)*10^(-0.1*MDFEXT(j)); 
   ct_fx(j) = (2*deltaF * sum_fx).^(0.5) ; % Far-end Crosstalk
end


%%%% Totla ICN %%%%%%
ct_total = (ct_nx.^2 + ct_fx.^2).^(0.5);


end

