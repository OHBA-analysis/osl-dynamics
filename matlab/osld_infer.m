function R = osld_infer(S)
% State time courses of data with a model trained by osld_train
% FORMAT R = osld_infer(S)
%
% S                    - options:
%   .model_dir           model folder (HMM.model_dir)             [required]
%   .data                cell array of [samples x channels] arrays, or file
%                        paths (.npy, .mat, .fif), one per session [required]
%   .data_field          variable in .mat files                        ['X']
%   .time_axis_first     the first dimension of the data is time      [true]
%   .sampling_frequency  Hz                                       [required]
%   .dual_estimation     estimate the means and covariances of each
%                        session                                     [false]
%   .n_jobs              parallel processes for dual estimation          [1]
%   .python              Python executable                    [osld_python]
%   .jobroot             folder for the temporary job folder     [tempdir]
%
% R                    - .alpha, .path, .stats as in osld_train; with
%                        dual estimation, .means {sessions} [C x K] and
%                        .covs {sessions} [C x C x K]
%
% The data are prepared as for training (osld_train's S.prepare), with the
% principal components of the training data.
%__________________________________________________________________________

for f = {'model_dir','data','sampling_frequency'}
    if ~isfield(S,f{1}) || isempty(S.(f{1})), error('osld:invalidInput','S.%s is required.',f{1}); end
end
def = struct('data_field','X','time_axis_first',true,'dual_estimation',false,'n_jobs',1, ...
    'python','','jobroot','');
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
d = S.data;
if isstring(d), d = cellstr(d); end
if ~iscell(d), d = {d}; end
for i = 1:numel(d)
    if isnumeric(d{i}), d{i} = single(d{i}); end
end

args = struct('model_dir',char(S.model_dir),'data',{d},'sampling_frequency',S.sampling_frequency, ...
    'dual_estimation',logical(S.dual_estimation),'n_jobs',S.n_jobs, ...
    'data_field',S.data_field,'time_axis_first',logical(S.time_axis_first));
R = osld_run('infer',args,struct('python',S.python,'jobroot',S.jobroot));
