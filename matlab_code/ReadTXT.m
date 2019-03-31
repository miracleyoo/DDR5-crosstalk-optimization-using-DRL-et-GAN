function [FE, NE, TT] = ReadTXT(filename1)

fileID1 = fopen(filename1);
C1 = textscan(fileID1,'%s %s %s %s');
for kk = 2: (length(C1{1}))
    FE(1,kk-1) = str2num(cell2mat( C1{2}(kk)));
    NE(1,kk-1)= str2num(cell2mat( C1{3}(kk)));
    TT(1,kk-1) = str2num(cell2mat( C1{4}(kk)));
end
fclose(fileID1);


end

