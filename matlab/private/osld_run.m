function out = osld_run(task,args,S)
% Run a task of python/osld_bridge.py and return its result
% FORMAT out = osld_run(task,args,S)
%
% task     - task name (see TASKS in osld_bridge.py)
% args     - struct of arguments: numeric arrays (single arrays are passed
%            as float32), cell arrays of them, scalars, logicals, character
%            vectors and structs of these; empty fields are left out
% S        - options:
%   .python  Python executable                                [osld_python]
%   .jobroot folder in which the job folder is made              [tempdir]
%
% out      - struct of the results: arrays are double, lists are cell
%            arrays
%
% The task runs in a new Python process; the arguments and results are
% exchanged through files in a job folder, which is deleted afterwards.
% The Python output (e.g. training progress) is shown in the Command
% Window.
%__________________________________________________________________________

if nargin < 2 || isempty(args), args = struct(); end
if nargin < 3, S = struct(); end
if ~isfield(S,'python') || isempty(S.python), S.python = osld_python(); end
if isempty(S.python)
    error('osld:noPython',['No Python with osl-dynamics found. Install it with ' ...
        'osld_install, or give an existing one with osld_python.']);
end
if ~isfield(S,'jobroot') || isempty(S.jobroot), S.jobroot = tempdir; end

%-Job folder with the arguments
%--------------------------------------------------------------------------
[~,name] = fileparts(tempname);
job = fullfile(S.jobroot,['osld_job_' name]);
mkdir(job);
cleanup = onCleanup(@() remove_dir(job));
n = 0;
enc = encode(args);
osld_write_text(fullfile(job,'job.json'),jsonencode(struct('task',task,'args',enc)));

%-Run
%--------------------------------------------------------------------------
cmd = sprintf('"%s" "%s" "%s"',S.python,osld_paths().bridge,job);
vars = {'PYTHONNOUSERSITE','1'; 'PYTHONDONTWRITEBYTECODE','1'; 'PYTHONPATH',''; ...
    'PYTHONHOME',''; 'PYTHONIOENCODING','utf-8'; 'PYTHONUNBUFFERED','1'; ...
    'TF_CPP_MIN_LOG_LEVEL','2'};
if ispc
    cmd = ['call ' cmd];                % a command starting with a quote is altered by cmd.exe
else
    vars = [vars; {'LD_LIBRARY_PATH',''; 'DYLD_LIBRARY_PATH',''}];   % MATLAB's libraries
end
restore = osld_setenv(vars); %#ok<NASGU> restored when cleared
status = system(cmd,'-echo');
clear restore
if status ~= 0
    error('osld:python','%s',failure_message(task,status,job));
end

%-Results
%--------------------------------------------------------------------------
out = decode(jsondecode(osld_read_text(fullfile(job,'result.json'))));


    function y = encode(x)
        if isstruct(x)
            y = struct();
            f = fieldnames(x);
            for i = 1:numel(f)
                v = x.(f{i});
                if isempty(v) && ~ischar(v) && ~iscell(v) && ~isstruct(v), continue; end
                y.(f{i}) = encode(v);
            end
        elseif iscell(x)
            y = cellfun(@encode,x,'UniformOutput',false);
            y = y(:)';
        elseif isstring(x)
            if isscalar(x), y = char(x); else, y = cellstr(x(:))'; end
        elseif (isnumeric(x) || islogical(x)) && ~isscalar(x)
            n = n + 1;
            file = sprintf('in%d.bin',n);
            if isa(x,'single'), dtype = 'float32'; else, dtype = 'float64'; end
            fid = fopen(fullfile(job,file),'w');
            fwrite(fid,x(:),dtype);
            fclose(fid);
            y = struct('ndarray',file,'shape',size(x),'dtype',dtype);
        elseif isnumeric(x)
            y = double(x);
        else
            y = x;
        end
    end

    function y = decode(x)
        if isstruct(x) && isscalar(x) && isfield(x,'ndarray')
            y = read_array(x);
        elseif isstruct(x) && isscalar(x) && isequal(fieldnames(x),{'list'})
            items = x.list;
            if isstruct(items), items = num2cell(items); end
            if ~iscell(items), items = num2cell(items); end
            y = cellfun(@decode,items(:),'UniformOutput',false);
        elseif isstruct(x)
            y = struct();
            f = fieldnames(x);
            for i = 1:numel(f), y.(f{i}) = decode(x.(f{i})); end
        elseif iscell(x)
            y = cellfun(@decode,x,'UniformOutput',false);
        else
            y = x;
        end
    end

    function a = read_array(r)
        sz  = double(r.shape(:)');
        fid = fopen(fullfile(job,r.ndarray),'r');
        a   = fread(fid,prod(sz),[r.dtype '=>double']);
        fclose(fid);
        if isscalar(sz), sz = [sz 1]; end
        a = reshape(a,sz);
    end
end


function msg = failure_message(task,status,job)
msg = sprintf('The osl-dynamics task ''%s'' failed (exit status %d).',task,status);
err = fullfile(job,'error.txt');
if exist(err,'file')
    lines = strsplit(strtrim(strrep(osld_read_text(err),char(13),'')),newline);
    msg = sprintf('%s\nPython traceback (last lines):\n%s',msg,strjoin(lines(max(1,end-14):end),newline));
    return
end
% Windows exit codes of a process that crashed before Python could report
switch mod(status,2^32)
    case hex2dec('C000001D')
        msg = [msg ' Illegal instruction: the CPU (or the x64 emulation on Windows on ARM) ' ...
            'lacks an instruction TensorFlow needs, e.g. AVX. On Windows on ARM, update ' ...
            'Windows 11 to 24H2 or later with the latest updates.'];
    case hex2dec('C0000135')
        msg = [msg ' A DLL was not found: install the Microsoft Visual C++ ' ...
            'Redistributable (x64), https://aka.ms/vs/17/release/vc_redist.x64.exe.'];
    case hex2dec('C0000142')
        msg = [msg ' A DLL failed to initialize.'];
    case hex2dec('C0000005')
        msg = [msg ' Access violation (the Python process crashed).'];
    otherwise
        msg = [msg ' See the Python output above.'];
end
end


function remove_dir(d)
if exist(d,'dir')
    [ok,m] = rmdir(d,'s');
    if ~ok, warning('osld:cleanup','Cannot delete the job folder %s: %s',d,m); end
end
end
