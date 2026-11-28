function R = osld_windows_test(action)
% Test the MATLAB interface of osl-dynamics on this computer
% FORMAT R = osld_windows_test               - automatic tests
% FORMAT osld_windows_test('ctrlc')          - Ctrl+C test (interactive)
% FORMAT osld_windows_test('processes')      - leftover Python processes
% FORMAT osld_windows_test('uninstall')      - remove the installation
%
% R        - .passed (all automatic tests passed), .stages (struct array:
%            .name, .passed, .seconds, .details, .error)
%
% The automatic tests collect information about the computer, install
% osl-dynamics with osld_install, import TensorFlow, train a small hidden
% Markov model on simulated data, infer the states of new data (with dual
% estimation in parallel), repeat the inference in folders with unusual
% names and check that no temporary files are left. The results are written
% to osld_windows_report.txt and the Command Window output to
% osld_windows_log.txt, next to this file. The user and computer names are
% replaced by <user> and <computer> in both files.
%__________________________________________________________________________

if nargin < 1, action = 'all'; end
kit    = fileparts(mfilename('fullpath'));
report = fullfile(kit,'osld_windows_report.txt');
logf   = fullfile(kit,'osld_windows_log.txt');

switch lower(action)
    case 'all'
        if exist(logf,'file'), delete(logf); end
        diary(logf);
        stop = onCleanup(@() diary('off'));
        R = run_all();
        clear stop
        osld_write_text(report,redact(format_report(R)));
        if exist(logf,'file'), osld_write_text(logf,redact(osld_read_text(logf))); end
        fprintf('\n%s\nReport: %s\nLog:    %s\n',summary_line(R),report,logf);
    case 'ctrlc'
        ctrlc_test(report,logf);
    case 'processes'
        txt = processes();
        disp(txt);
        osld_write_text(report,redact(txt),'a');
        fprintf('Added to %s\n',report);
    case 'uninstall'
        d = osld_paths(osld_settings_dir()).dir;
        osld_install(struct('uninstall',true,'confirm',false,'dir',d));
        osld_write_text(report,redact(sprintf(['\n== Uninstall (%s) ==\n%s exists afterwards: ' ...
            '%d\n'],char(datetime('now')),d,exist(d,'dir') == 7)),'a');
    otherwise
        error('osld:invalidInput','Unknown action ''%s''.',action);
end
if nargout == 0, clear R; end
end


%==========================================================================
function R = run_all()
K = 3; C = 8; fs = 250; L = 200;
n_train = [20100 20100]; n_test = 10050;
cfg = struct('n_states',K,'sequence_length',L,'batch_size',16,'learning_rate',0.01, ...
    'n_epochs',10,'n_init',3,'n_init_epochs',1);
[X,Z,Xt,Zt] = simulate(K,C,n_train,n_test);
work = fullfile(tempdir,'osld_windows_test');
started = now;
if exist(work,'dir'), rmdir(work,'s'); end
mkdir(work);
model_dir = '';
mapping = 1:K;

stages = struct('name',{},'passed',{},'seconds',{},'details',{},'error',{});
stages(end+1) = stage('System information',@() deal(true,system_info()));
stages(end+1) = stage('Install osl-dynamics',@install);
if stages(end).passed
    stages(end+1) = stage('Python and TensorFlow',@python_check);
end
if stages(end).passed
    stages(end+1) = stage('Train an HMM on simulated data',@train);
end
if stages(end).passed
    stages(end+1) = stage('Infer new data, dual estimation with n_jobs = 2',@infer);
    stages(end+1) = stage('Folder with spaces and accented letters', ...
        @() odd_folder(['osld test ' char([228 32 233])]));
    stages(end+1) = stage('Folder with a Greek letter',@() odd_folder(['osld_' char(945)]));
    stages(end+1) = stage('Temporary files removed',@cleanup_check);
end
R = struct('passed',all([stages.passed]),'stages',stages);

    function [ok,d] = install()
        info = osld_install();
        d = info;
        ok = exist(info.python,'file') == 2;
    end

    function [ok,d] = python_check()
        d = osld_run('versions',struct('check',true));
        ok = ~isempty(d.tensorflow);
    end

    function [ok,d] = train()
        S = struct('data',{X},'sampling_frequency',fs,'config',cfg, ...
            'prepare',struct('standardize',struct()),'outdir',fullfile(work,'hmm'),'seed',1);
        H = osld_train(S);
        model_dir = H.model_dir;
        [acc,mapping] = accuracy(H.path,Z,K,[]);
        aligned = check_alignment(H.alpha,H.path,n_train,L);
        n_steps = cfg.n_epochs + cfg.n_init*cfg.n_init_epochs;
        d = struct('accuracy',acc,'aligned',aligned,'free_energy',H.free_energy, ...
            'train_seconds',H.train_seconds,'seconds_per_epoch',H.train_seconds/n_steps, ...
            'fractional_occupancy',H.stats.fractional_occupancy);
        ok = all(acc > 0.9) && aligned;
    end

    function [ok,d] = infer()
        S = struct('model_dir',model_dir,'data',{{Xt,X{1}}},'sampling_frequency',fs, ...
            'dual_estimation',true,'n_jobs',2);
        r = osld_infer(S);
        acc = accuracy(r.path,{Zt,Z{1}},K,mapping);
        aligned = check_alignment(r.alpha,r.path,[n_test n_train(1)],L);
        sizes = numel(r.means) == 2 && all(cellfun(@(m) isequal(size(m),[C K]),r.means)) ...
            && all(cellfun(@(c) isequal(size(c),[C C K]),r.covs));
        d = struct('accuracy',acc,'aligned',aligned,'dual_estimation_sizes',sizes, ...
            'dual_estimation_seconds',r.dual_estimation_seconds);
        ok = all(acc > 0.9) && aligned && sizes;
    end

    function [ok,d] = odd_folder(name)
        folder = fullfile(work,name);
        mkdir(folder);
        file = fullfile(folder,'session 1.mat');
        saved = struct('X',single(Xt));
        save(file,'-struct','saved','-v7');
        S = struct('model_dir',model_dir,'data',file,'sampling_frequency',fs,'jobroot',folder);
        r = osld_infer(S);
        acc = accuracy(r.path,{Zt},K,mapping);
        d = struct('folder',folder,'accuracy',acc);
        ok = acc > 0.9;
    end

    function [ok,d] = cleanup_check()
        left = [dir(fullfile(tempdir,'osld_job_*')); dir(fullfile(work,'**','osld_job_*'))];
        left = left([left.datenum] >= started - 1/86400);  % made during this run
        d = struct('leftover_job_folders',{fullfile({left.folder},{left.name})});
        ok = isempty(left);
        [removed,msg] = rmdir(work,'s');
        d.work_folder_removed = removed;
        if ~removed, d.work_folder_error = msg; end
    end
end


function st = stage(name,fun)
fprintf('\n=== %s ===\n',name);
st = struct('name',name,'passed',false,'seconds',0,'details',struct(),'error','');
t = tic;
try
    [st.passed,st.details] = fun();
catch err
    st.error = getReport(err,'extended','hyperlinks','off');
    fprintf(2,'%s\n',st.error);
end
st.seconds = toc(t);
fprintf('[%s] %s (%.1f s)\n',pass_fail(st.passed),name,st.seconds);
end


function [X,Z,Xt,Zt] = simulate(K,C,n_train,n_test)
% Zero-mean Gaussian states with different covariances and a Markov chain
s = rng(1);
restore = onCleanup(@() rng(s));
A = 0.98*eye(K) + 0.02/(K - 1)*(1 - eye(K));
R = cell(K,1);
for k = 1:K
    [Q,~] = qr(randn(C));
    R{k} = chol(Q*diag(linspace(0.2,3,C).^(k/2 + 0.5))*Q' + 0.05*eye(C));
end
X = cell(numel(n_train),1); Z = X;
for i = 1:numel(n_train), [X{i},Z{i}] = one(n_train(i)); end
[Xt,Zt] = one(n_test);

    function [x,z] = one(n)
        z = zeros(n,1); z(1) = 1;
        cA = cumsum(A,2);
        u = rand(n,1);
        for t = 2:n, z(t) = find(u(t) < cA(z(t-1),:),1); end
        x = zeros(n,C,'single');
        for q = 1:K
            j = z == q;
            x(j,:) = randn(nnz(j),C)*R{q};
        end
    end
end


function [acc,mapping] = accuracy(path,Z,K,mapping)
% Agreement of the Viterbi paths with the true states (best relabeling)
if isempty(mapping)
    P = perms(1:K);
    best = -1;
    for i = 1:size(P,1)
        a = agree(path{1},Z{1},P(i,:));
        if a > best, best = a; mapping = P(i,:); end
    end
end
acc = cellfun(@(p,z) agree(p,z,mapping),path(:)',Z(:)');

    function a = agree(p,z,m)
        v = ~isnan(p);
        a = mean(m(p(v))' == z(v));
    end
end


function ok = check_alignment(alpha,path,n,L)
% A row for every sample; NaN only after the last whole sequence
ok = true;
for i = 1:numel(alpha)
    nan_rows = mod(n(min(i,end)),L);
    ok = ok && size(alpha{i},1) == n(min(i,end)) && numel(path{i}) == n(min(i,end)) ...
        && nnz(isnan(path{i})) == nan_rows && all(isnan(path{i}(end-nan_rows+1:end))) ...
        && max(abs(sum(alpha{i}(1:end-nan_rows,:),2) - 1)) < 1e-3;
end
end


%==========================================================================
function d = system_info()
d.date = char(datetime('now','Format','yyyy-MM-dd HH:mm'));
d.matlab = struct('version',version,'arch',computer('arch'),'cores',feature('numcores'));
if ~ispc
    [~,u] = system('uname -a');
    d.os = strtrim(u);
    d.home = getenv('HOME');
    return
end
nt = 'SOFTWARE\Microsoft\Windows NT\CurrentVersion';
d.os = struct('product',reg('HKEY_LOCAL_MACHINE',nt,'ProductName'), ...
    'display_version',reg('HKEY_LOCAL_MACHINE',nt,'DisplayVersion'), ...
    'build',reg('HKEY_LOCAL_MACHINE',nt,'CurrentBuild'),'ubr',reg('HKEY_LOCAL_MACHINE',nt,'UBR'));
d.os.note = 'Windows 11 has build 22000 or later (ProductName may still say Windows 10)';
env_key = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
d.cpu = struct('name',strtrim(char(reg('HKEY_LOCAL_MACHINE', ...
    'HARDWARE\DESCRIPTION\System\CentralProcessor\0','ProcessorNameString'))), ...
    'architecture_registry',reg('HKEY_LOCAL_MACHINE',env_key,'PROCESSOR_ARCHITECTURE'), ...
    'architecture_matlab_env',getenv('PROCESSOR_ARCHITECTURE'), ...
    'architecture_w6432',getenv('PROCESSOR_ARCHITEW6432'), ...
    'identifier',getenv('PROCESSOR_IDENTIFIER'), ...
    'architecture_cmd',shell('echo %PROCESSOR_ARCHITECTURE%'), ...
    'architecture_powershell',shell(['powershell -NoProfile -Command "' ...
        '[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture"']));
d.arm = any(contains({char(d.cpu.architecture_registry),d.cpu.architecture_powershell, ...
    d.cpu.identifier,d.cpu.name},{'ARM','Arm','Snapdragon'}));
d.long_paths_enabled = reg('HKEY_LOCAL_MACHINE','SYSTEM\CurrentControlSet\Control\FileSystem','LongPathsEnabled');
d.elevated = system('net session >nul 2>&1') == 0;
vc = 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64';
sys32 = fullfile(getenv('SystemRoot'),'System32');
d.vc_runtime = struct('version',reg('HKEY_LOCAL_MACHINE',vc,'Version'), ...
    'msvcp140',exist(fullfile(sys32,'msvcp140.dll'),'file') == 2, ...
    'vcruntime140',exist(fullfile(sys32,'vcruntime140.dll'),'file') == 2, ...
    'vcruntime140_1',exist(fullfile(sys32,'vcruntime140_1.dll'),'file') == 2);
f = struct('userprofile',getenv('USERPROFILE'),'localappdata',getenv('LOCALAPPDATA'), ...
    'temp',tempdir,'onedrive',getenv('OneDrive'),'homeshare',getenv('HOMESHARE'), ...
    'install_dir',osld_paths().default_dir);
f.non_ascii = any(cellfun(@(s) any(double(s) > 127),struct2cell(f)));
f.spaces = any(contains({f.userprofile,f.localappdata,f.temp},' '));
f.localappdata_in_onedrive = ~isempty(f.onedrive) && startsWith(f.localappdata,f.onedrive,'IgnoreCase',true);
d.folders = f;
d.free_space_gb = str2double(shell(['powershell -NoProfile -Command "(Get-PSDrive -Name ' ...
    '(Get-Item -LiteralPath $env:LOCALAPPDATA).PSDrive.Name).Free"']))/1e9;
inet = 'Software\Microsoft\Windows\CurrentVersion\Internet Settings';
d.proxy = struct('https_proxy',getenv('HTTPS_PROXY'),'http_proxy',getenv('HTTP_PROXY'), ...
    'no_proxy',getenv('NO_PROXY'),'enabled',reg('HKEY_CURRENT_USER',inet,'ProxyEnable'), ...
    'server',reg('HKEY_CURRENT_USER',inet,'ProxyServer'), ...
    'auto_config_url',reg('HKEY_CURRENT_USER',inet,'AutoConfigURL'), ...
    'winhttp',shell('netsh winhttp show proxy'));
end


function v = reg(hive,key,name)
try
    v = winqueryreg(hive,key,name);
catch
    v = '';
end
end


function out = shell(cmd)
[status,out] = system(cmd);
out = strtrim(out);
if status ~= 0, out = sprintf('(exit status %d) %s',status,out); end
end


%==========================================================================
function ctrlc_test(report,logf)
diary(logf);
fprintf(['\nCtrl+C test: Python will run for 2 minutes. When lines starting with ' ...
    '"busy:" appear,\npress Ctrl+C in the Command Window. Afterwards run\n' ...
    '    osld_windows_test(''processes'')\n\n']);
osld_write_text(report,sprintf('\n== Ctrl+C test ==\nStarted: %s\n',char(datetime('now'))),'a');
t = tic;
done = onCleanup(@() ctrlc_end(report,t));
r = osld_run('busy',struct('seconds',120));
osld_write_text(report,sprintf('Finished without interruption (Python process %d)\n',r.pid),'a');
end


function ctrlc_end(report,t)
osld_write_text(report,sprintf('Ended: %s (after %.0f s)\n',char(datetime('now')),toc(t)),'a');
diary('off');
end


function txt = processes()
env = osld_paths(osld_settings_dir()).env;
txt = sprintf('\n== Python processes of %s (%s) ==\n',env,char(datetime('now')));
if ispc
    cmd = sprintf(['powershell -NoProfile -Command "Get-Process python -ErrorAction ' ...
        'SilentlyContinue | Where-Object { $_.Path -like ''%s*'' } | Select-Object Id,Path,' ...
        'StartTime | Format-Table -AutoSize | Out-String -Width 300"'],strrep(env,'''',''''''));
else
    cmd = sprintf('ps -ax -o pid,command | grep -F "%s" | grep -v grep',env);
end
[~,out] = system(cmd);
out = strtrim(out);
if isempty(out), out = 'None'; end
txt = sprintf('%s%s\n',txt,out);
end


function d = osld_settings_dir()
s = osld_settings();
if isfield(s,'dir'), d = s.dir; else, d = ''; end
end


%==========================================================================
function txt = format_report(R)
lines = {'osl-dynamics MATLAB interface: Windows test report', summary_line(R), ''};
for i = 1:numel(R.stages)
    st = R.stages(i);
    lines{end+1} = sprintf('[%s] %s (%.1f s)',pass_fail(st.passed),st.name,st.seconds); %#ok<AGROW>
    lines = [lines, format_value(st.details,'    ')]; %#ok<AGROW>
    if ~isempty(st.error)
        lines = [lines, strcat({'    ! '},strsplit(st.error,newline))]; %#ok<AGROW>
    end
    lines{end+1} = ''; %#ok<AGROW>
end
txt = strjoin(lines,newline);
end


function lines = format_value(v,indent)
lines = {};
if ~isstruct(v), lines = {[indent value_text(v)]}; return; end
f = fieldnames(v);
for i = 1:numel(f)
    x = v.(f{i});
    if isstruct(x) && isscalar(x)
        lines{end+1} = sprintf('%s%s:',indent,f{i}); %#ok<AGROW>
        lines = [lines, format_value(x,[indent '    '])]; %#ok<AGROW>
    else
        t = value_text(x);
        if contains(t,newline)
            lines{end+1} = sprintf('%s%s:',indent,f{i}); %#ok<AGROW>
            lines = [lines, strcat({[indent '    ']},strsplit(t,newline))]; %#ok<AGROW>
        else
            lines{end+1} = sprintf('%s%s: %s',indent,f{i},t); %#ok<AGROW>
        end
    end
end
end


function t = value_text(x)
if ischar(x)
    t = x;
elseif isstring(x)
    t = char(strjoin(x,', '));
elseif islogical(x) && isscalar(x)
    t = mat2str(x);
elseif isnumeric(x) && isscalar(x)
    t = num2str(x,6);
elseif isnumeric(x) || islogical(x)
    t = mat2str(x,4);
elseif iscell(x)
    t = strjoin(cellfun(@value_text,x,'UniformOutput',false),newline);
    if isempty(x), t = '(none)'; end
elseif isstruct(x)
    t = sprintf('[%s struct]',mat2str(size(x)));
else
    t = class(x);
end
end


function s = summary_line(R)
s = sprintf('%d of %d stages passed (%s)',nnz([R.stages.passed]),numel(R.stages), ...
    char(datetime('now','Format','yyyy-MM-dd HH:mm')));
end


function s = pass_fail(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end


function txt = redact(txt)
names = {getenv('USERNAME'),'<user>'; getenv('USER'),'<user>'; getenv('COMPUTERNAME'),'<computer>'};
for i = 1:size(names,1)
    if numel(names{i,1}) >= 3
        txt = regexprep(txt,regexptranslate('escape',names{i,1}),names{i,2},'ignorecase');
    end
end
end
