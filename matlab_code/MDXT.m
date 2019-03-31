function [MDNEXT, MDFEXT] = MDXT(sdd, vic, ptOrder)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %%%% aggressors identification %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    portNum = min(size(sdd));
    
    %%%%%  THU: 1->n+1  %%%%%%
    if(ptOrder==2)
        k = 1;
        for i = 1: (0.5*portNum)
            if(vic ~= i)
                NE_agg(k) = i;
                FE_agg(k) = i+(0.5*portNum);
                k = k+1;
            end
        end

    end
    
    %%%%%%  THU: 1->2  %%%%%%
    if(ptOrder==1)
        k = 1;
        for i = 1: (0.5*portNum)
            if(vic ~= (2*i-1) )
                NE_agg(k) = 2*i-1;
                FE_agg(k) = 2*i;
                k = k+1;
            end
        end
    end
    
    % disp('NE_AGG:'); disp(NE_agg);
    % disp('FE_AGG:'); disp(FE_agg);  
    % disp(' ')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %%%% Get all the NE and FE %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for kk = 1:length(NE_agg) 
        for ii = 1: length(sdd)
         % ALL THE FAR-END   
         NE(kk, ii) = sdd(vic, NE_agg(kk), ii);
         % ALL THE FAR-END
         FE(kk, ii) = sdd(vic, FE_agg(kk), ii);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %%%% Get the accunulative NE&FE  %%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SUM_NE = 0;
    SUM_FE = 0;
    
    for jj = 1:length(NE_agg) 
        SUM_NE = SUM_NE + abs(NE(jj,:)).^2;
        SUM_FE = SUM_FE + abs(FE(jj,:)).^2;
    end
    
    MDNEXT = -10* log10(SUM_NE);
    MDFEXT = -10* log10(SUM_FE);
    
%     figure(); plot(MDNEXT, 'r');
%     figure(); plot(MDFEXT, 'b');
end

