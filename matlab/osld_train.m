function HMM = osld_train(S)
% Train a hidden Markov model with osl-dynamics
% FORMAT HMM = osld_train(S)
%
% S                    - options:
%   .data                cell array of [samples x channels] arrays, or file
%                        paths (.npy, .mat, .fif), one per session [required]
%   .data_field          variable in .mat files                        ['X']
%   .time_axis_first     the first dimension of the data is time      [true]
%   .sampling_frequency  Hz                                       [required]
%   .prepare             struct of osl_dynamics.data.Data methods and their
%                        options, applied in order, e.g.
%                        struct('tde_pca',struct('n_embeddings',15, ...
%                        'n_pca_components',80),'standardize',struct())
%                                                                 [struct()]
%   .config              options of osl_dynamics.models.hmm.Config; n_states
%                        is required. Defaults: sequence_length 200,
%                        batch_size 256, learning_rate 0.01, n_epochs 20,
%                        learn_means false, learn_covariances true
%   .outdir              folder for the trained model             [required]
%   .seed                random seed                                   [[]]
%   .python              Python executable                    [osld_python]
%
% HMM                  - .model_dir, .config, .free_energy,
%                        .initial_state_probs [K x 1], .trans_prob [K x K],
%                        .means [C x K], .covs [C x C x K] (in the space of
%                        the prepared data), .alpha {sessions} [samples x K],
%                        .path {sessions} [samples x 1] (states 1..K),
%                        .stats [sessions x K]: .fractional_occupancy,
%                        .mean_lifetime, .mean_interval (s), .switching_rate
%                        (per s), .pca_components (with PCA), .train_seconds
%
% .alpha and .path have a row for every input sample; samples without a
% state (lost to time-delay embedding or after the last whole sequence)
% are NaN.
%__________________________________________________________________________

S = defaults(S,{'data','sampling_frequency','config','outdir'}, ...
    struct('data_field','X','time_axis_first',true,'prepare',struct(),'seed',[],'python',''));
if ~isfield(S.config,'n_states'), error('osld:invalidInput','S.config.n_states is required.'); end
outdir = char(S.outdir);
if ~exist(outdir,'dir'), mkdir(outdir); end
model_dir = fullfile(outdir,'model');

args = struct('data',{data_arg(S.data)},'sampling_frequency',S.sampling_frequency, ...
    'config',S.config,'model_dir',model_dir,'prepare',S.prepare,'seed',S.seed, ...
    'data_field',S.data_field,'time_axis_first',logical(S.time_axis_first));
r = osld_run('train_hmm',args,struct('python',S.python));

HMM = struct('model_dir',model_dir,'config',r.config,'free_energy',r.free_energy, ...
    'initial_state_probs',r.initial_state_probs(:),'trans_prob',r.trans_prob, ...
    'means',r.means,'covs',r.covs,'alpha',{r.alpha},'path',{r.path},'stats',r.stats, ...
    'train_seconds',r.train_seconds);
if isfield(r,'pca_components'), HMM.pca_components = r.pca_components; end


function S = defaults(S,required,def)
for i = 1:numel(required)
    if ~isfield(S,required{i}) || isempty(S.(required{i}))
        error('osld:invalidInput','S.%s is required.',required{i});
    end
end
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end


function d = data_arg(d)
% Cell array of single arrays or file paths
if isstring(d), d = cellstr(d); end
if ~iscell(d), d = {d}; end
for i = 1:numel(d)
    if isnumeric(d{i}), d{i} = single(d{i}); end
end
