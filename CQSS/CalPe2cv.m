function [xe_pe, xe_pe_mean,xe_pe_std] = CalPe2cv(s_amp)
%****************************************************************
%求变异系数
s_amp_mean = mean(s_amp, 2);
s_amp_std = std(s_amp,1, 2);
 xe_pe =  s_amp_std./s_amp_mean; 
%****************************************************************

xe_pe_mean =log2(mean(s_amp, 2));
xe_pe_std =log2(std(s_amp,1, 2));
