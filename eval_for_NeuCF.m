function [ rate ] = eval_for_NeuCF( metric, topR )
% metric : evaluation for ScoreMatrix
%           1 indicates cdf, 0 indicates recall

load('NeuCF_ScoreMatrix.mat');
load('splits_uniform.mat');
T = full(splits{1});
if metric
   rate = cdf(T, ScoreMatrix, topR);
else
   rate = recall(T, ScoreMatrix, topR);
end

end

