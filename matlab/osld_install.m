function info = osld_install(S)
% Install osl-dynamics in a Python environment for the MATLAB interface
% FORMAT info = osld_install(S)
% FORMAT osld_install('uninstall')
%
% S             - options:
%   .dir          installation folder (about 1.5 GB)
%                 [%LOCALAPPDATA%\osl-dynamics\matlab on Windows,
%                  ~/.osl-dynamics/matlab elsewhere]
%   .python       Python version                                    ['3.12']
%   .proxy        proxy for the downloads by uv, e.g.
%                 'http://proxy.example.ac.uk:8080'                    ['']
%   .system_certs uv uses the operating system's certificates (needed
%                 behind proxies that inspect HTTPS)  [true on Windows]
%   .source       local osl-dynamics checkout to install (editable)
%                 instead of the version from PyPI                     ['']
%   .force        reinstall even if osl-dynamics is installed        [false]
%   .confirm      ask before uninstalling                             [true]
%
% info          - .python (executable), .dir, .osl_dynamics, .tensorflow,
%                 .seconds (time of each step)
%
% Downloads uv (https://docs.astral.sh/uv), which installs Python and then
% osl-dynamics and TensorFlow from PyPI into a new environment in S.dir:
% osl-dynamics OSLD_VERSION, with every other package at its latest version
% released before EXCLUDE_NEWER. Dependencies the MATLAB interface does not
% use (EXCLUDES) are left out. On Windows, the x64 Python is installed, which
% runs under emulation on Windows on ARM (TensorFlow has no Windows ARM
% build). The environment is recorded in the settings file, where
% osld_python finds it. Needs an internet connection; nothing outside S.dir
% and the settings file is changed.
%__________________________________________________________________________

OSLD_VERSION  = '3.9.1';
EXCLUDE_NEWER = '2026-11-27';
UV_VERSION    = '0.13.10';
EXCLUDES      = {'fslpy','h5io','ipyevents','ipywidgets','mne-icalabel', ...
    'nest-asyncio2','numba','onnxruntime','osfclient','pyside6','pyvista', ...
    'pyvistaqt','scikit-image','trame','trame-vtk','trame-vuetify','vtk'};
PLATFORMS     = { ...  % MATLAB arch, uv archive, SHA-256, uv Python platform
    'maca64',  'uv-aarch64-apple-darwin.tar.gz',     'e419bf07533bdbf48795fc793601c834dd79db57a8c49ce8e25cfe3a8639958d', 'macos-aarch64-none'
    'maci64',  'uv-x86_64-apple-darwin.tar.gz',      '153e03197588204c04fc4c81cc4bafbfa4280222a385e4ee36a87aeec0ae63b4', 'macos-x86_64-none'
    'glnxa64', 'uv-x86_64-unknown-linux-musl.tar.gz','5597970380083a340b7983af05a84824b4db1c62b7059c796935b7fd6dc95fd2', 'linux-x86_64-gnu'
    'win64',   'uv-x86_64-pc-windows-msvc.zip',      '297b3881fe1adfb922becc2c1613202abb41dc83ea5ff43d472aa087ca1f8683', 'windows-x86_64-none'};

if nargin < 1, S = struct(); end
if ischar(S) && strcmpi(S,'uninstall'), S = struct('uninstall',true); end
def = struct('dir','','python','3.12','proxy','','system_certs',ispc,'source','', ...
    'force',false,'confirm',true,'uninstall',false);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(S,fn{i}), S.(fn{i}) = def.(fn{i}); end
end
if isempty(S.dir)
    s = osld_settings();
    if isfield(s,'dir') && ~isempty(s.dir), S.dir = s.dir; end
end
p = osld_paths(S.dir);

if S.uninstall
    uninstall(p,S.confirm);
    info = [];
    return
end

%-Already installed?
%--------------------------------------------------------------------------
if exist(p.python,'file') && ~S.force && isempty(S.source)
    v = try_versions(p.python);
    if ~isempty(v) && strcmp(v.osl_dynamics,OSLD_VERSION)
        fprintf('osl-dynamics %s is already installed in %s.\n',OSLD_VERSION,p.dir);
        info = finish(p,v,struct());
        return
    end
end

%-Checks
%--------------------------------------------------------------------------
arch = computer('arch');
a = find(strcmp(PLATFORMS(:,1),arch));
if isempty(a)
    error('osld:install','osld_install does not support this platform (%s).',arch);
end
if ~exist(p.dir,'dir'), mkdir(p.dir); end
check_space(p.dir,3e9);
if ispc, check_path_length(p.env); end
seconds = struct();

%-uv
%--------------------------------------------------------------------------
t = tic;
if ~exist(p.uv,'file') || ~contains(command_output(p.uv,'--version'),UV_VERSION)
    fprintf('Downloading uv %s...\n',UV_VERSION);
    if ~exist(p.bin,'dir'), mkdir(p.bin); end
    archive = fullfile(p.dir,PLATFORMS{a,2});
    url = sprintf('https://github.com/astral-sh/uv/releases/download/%s/%s',UV_VERSION,PLATFORMS{a,2});
    try
        websave(archive,url,weboptions('Timeout',120));
    catch err
        error('osld:download',['Cannot download %s (%s). Check the internet connection ' ...
            'and MATLAB''s proxy settings (Preferences > Web).'],url,err.message);
    end
    if ~strcmpi(sha256(archive),PLATFORMS{a,3})
        delete(archive);
        error('osld:install','The downloaded uv does not match its SHA-256 checksum.');
    end
    tmp = fullfile(p.dir,'uv_extract');
    if endsWith(archive,'.zip'), unzip(archive,tmp); else, untar(archive,tmp); end
    [~,name,ext] = fileparts(p.uv);
    exe = dir(fullfile(tmp,'**',[name ext]));
    movefile(fullfile(exe(1).folder,exe(1).name),p.uv,'f');
    rmdir(tmp,'s');
    delete(archive);
    if ~ispc, system(sprintf('chmod +x "%s"',p.uv)); end
end
% Run uv once: programs in user folders may be blocked (e.g. AppLocker)
[status,out] = system(quote_cmd(p.uv,'--version'));
if status ~= 0
    error('osld:blocked',['Cannot run %s (exit status %d):\n%s\nPrograms in this folder may be ' ...
        'blocked by your organization. Choose another folder with S.dir, or use an existing ' ...
        'Python environment with osld_python.'],p.uv,status,strtrim(out));
end
seconds.uv = toc(t);

%-Python environment and packages
%--------------------------------------------------------------------------
env = {'UV_CACHE_DIR',p.cache; 'UV_PYTHON_INSTALL_DIR',p.python_install; ...
    'UV_NO_CONFIG','1'; 'NO_COLOR','1'; 'UV_NO_PROGRESS','1'; 'UV_LINK_MODE','copy'; ...
    'VIRTUAL_ENV',''; 'PYTHONPATH',''; 'PYTHONHOME',''};
if S.system_certs, env = [env; {'UV_SYSTEM_CERTS','1'}]; end
if ~isempty(S.proxy), env = [env; {'HTTPS_PROXY',S.proxy; 'HTTP_PROXY',S.proxy}]; end
if exist(p.env,'dir'), rmdir(p.env,'s'); end
t = tic;
request = sprintf('cpython-%s-%s',S.python,PLATFORMS{a,4});
fprintf('Installing Python %s (%s) in %s...\n',S.python,request,p.env);
run(p.uv,sprintf('venv --python %s --managed-python "%s"',request,p.env),env);
seconds.python = toc(t);

t = tic;
excludes = [tempname '.txt'];
osld_write_text(excludes,sprintf('%s\n',EXCLUDES{:}));
cleanup = onCleanup(@() delete(excludes));
if isempty(S.source)
    fprintf('Installing osl-dynamics %s and its dependencies...\n',OSLD_VERSION);
    run(p.uv,sprintf(['pip install --python "%s" --exclude-newer %s ' ...
        '--exclude-newer-package osl-dynamics=%s --excludes "%s" "osl-dynamics[tf]==%s"'], ...
        p.python,EXCLUDE_NEWER,char(datetime('now') + 1,'yyyy-MM-dd'),excludes,OSLD_VERSION),env);
else
    fprintf('Installing osl-dynamics from %s and its dependencies...\n',S.source);
    run(p.uv,sprintf('pip install --python "%s" --excludes "%s" -e "%s[tf]"', ...
        p.python,excludes,S.source),env);
end
clear cleanup
if exist(p.cache,'dir'), rmdir(p.cache,'s'); end
seconds.packages = toc(t);

%-Check and record
%--------------------------------------------------------------------------
t = tic;
v = osld_run('versions',struct('check',true),struct('python',p.python));
seconds.check = toc(t);
info = finish(p,v,seconds);
fprintf('Installed osl-dynamics %s (TensorFlow %s, Python %s) in %s.\n', ...
    v.osl_dynamics,v.tensorflow,v.python,p.dir);


%==========================================================================
function info = finish(p,v,seconds)
osld_settings(struct('python',p.python,'dir',p.dir,'osl_dynamics',v.osl_dynamics));
info = struct('python',p.python,'dir',p.dir,'osl_dynamics',v.osl_dynamics, ...
    'tensorflow',v.tensorflow,'seconds',seconds);


function v = try_versions(python)
try
    v = osld_run('versions',struct(),struct('python',python));
catch
    v = [];
end


function cmd = quote_cmd(exe,args)
cmd = sprintf('"%s" %s',exe,args);
if ispc, cmd = ['call ' cmd]; end    % a command starting with a quote is altered by cmd.exe


function run(uv,args,env)
restore = osld_setenv(env); %#ok<NASGU> restored when cleared
cmd = quote_cmd(uv,args);
status = system(cmd,'-echo');
if status ~= 0
    error('osld:install','The command failed (exit status %d): %s',status,cmd);
end


function out = command_output(exe,args)
[status,out] = system(quote_cmd(exe,args));
if status ~= 0, out = ''; end


function h = sha256(file)
% SHA-256 checksum with certutil or PowerShell (Windows), shasum (macOS) or
% sha256sum (Linux)
if ispc
    cmds = {sprintf('certutil -hashfile "%s" SHA256',file), sprintf(['powershell ' ...
        '-NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath ''%s'').Hash"'],file)};
elseif ismac
    cmds = {sprintf('shasum -a 256 "%s"',file)};
else
    cmds = {sprintf('sha256sum "%s"',file)};
end
for i = 1:numel(cmds)
    [status,out] = system(cmds{i});
    h = regexp(out,'^\s*([0-9a-fA-F]{2}\s?){32}','match','once','lineanchors');
    h = lower(regexprep(h,'\s',''));
    if status == 0 && numel(h) == 64, return; end
end
error('osld:install','Cannot compute the SHA-256 checksum of %s:\n%s',file,out);


function check_space(d,need)
% Free space from df (macOS, Linux) or PowerShell (Windows), if available
if ispc
    [status,out] = system(sprintf(['powershell -NoProfile -Command "(Get-PSDrive ' ...
        '-Name (Get-Item -LiteralPath ''%s'').PSDrive.Name).Free"'],d));
    free = str2double(strtrim(out));
else
    [status,out] = system(sprintf('df -Pk "%s" | tail -1',d));
    f = strsplit(strtrim(out));
    free = 1024*str2double(f{min(4,end)});
end
if status ~= 0 || isnan(free), return; end
if free < need
    error('osld:install',['%s has %.1f GB free; the installation needs about ' ...
        '%.0f GB. Choose another folder with S.dir.'],d,free/1e9,need/1e9);
end


function check_path_length(env)
% Windows limits paths to 260 characters unless long paths are enabled;
% the longest file in TensorFlow's package is about 160 characters below
% site-packages
n = numel(fullfile(env,'Lib','site-packages')) + 1 + 160;
if n < 260, return; end
try
    on = winqueryreg('HKEY_LOCAL_MACHINE','SYSTEM\CurrentControlSet\Control\FileSystem','LongPathsEnabled');
catch
    on = 0;
end
if ~on
    error('osld:install',['The installation folder %s is too long for Windows paths ' ...
        '(260 characters). Choose a shorter folder with S.dir, e.g. C:\\osl-dynamics.'],env);
end


function uninstall(p,confirm)
if ~exist(p.dir,'dir')
    fprintf('There is no installation in %s.\n',p.dir);
else
    if confirm
        a = input(sprintf('Delete the osl-dynamics installation in %s? [y/N] ',p.dir),'s');
        if ~strcmpi(strtrim(a),'y'), return; end
    end
    for d = {p.env,p.python_install,p.bin,p.cache}
        if exist(d{1},'dir'), rmdir(d{1},'s'); end
    end
    left = dir(p.dir);
    if isempty(setdiff({left.name},{'.','..'})), rmdir(p.dir); end
    fprintf('Deleted the installation in %s.\n',p.dir);
end
s = osld_settings();
if isfield(s,'dir') && strcmp(s.dir,p.dir) && exist(p.settings,'file')
    delete(p.settings);
end
