function [python,info] = osld_python(S)
% Python environment with osl-dynamics used by the MATLAB interface
% FORMAT [python,info] = osld_python
% FORMAT osld_python(S)
%
% S          - options:
%   .python    Python executable to use from now on, e.g. of an existing
%              environment with osl-dynamics; recorded in the settings
%              file                                                    ['']
%
% python     - Python executable, or '' if none is found
% info       - .source     where it was found: 'OSLD_PYTHON', 'settings' or
%                          'default'
%              .settings   settings file
%
% The Python is, in order: the one in the environment variable OSLD_PYTHON;
% the one recorded in the settings file (by osld_install or S.python); that
% of the default installation of osld_install. Without output arguments,
% the Python and the versions of the packages are printed.
%__________________________________________________________________________

if nargin < 1, S = struct(); end
p = osld_paths();
info.settings = p.settings;

%-Record a Python
%--------------------------------------------------------------------------
if isfield(S,'python') && ~isempty(S.python)
    if ~exist(S.python,'file')
        error('osld:invalidInput','%s does not exist.',S.python);
    end
    v = osld_run('versions',struct('check',true),struct('python',S.python));
    if isempty(v.osl_dynamics)
        error('osld:invalidInput','osl-dynamics is not installed in %s.',S.python);
    end
    osld_settings(struct('python',S.python));
    fprintf('osld_python: using %s (osl-dynamics %s).\n',S.python,v.osl_dynamics);
end

%-Find the Python
%--------------------------------------------------------------------------
python = getenv('OSLD_PYTHON'); info.source = 'OSLD_PYTHON';
if isempty(python)
    s = osld_settings();
    if isfield(s,'python') && ~isempty(s.python)
        python = s.python; info.source = 'settings';
    end
end
if isempty(python) && exist(p.python,'file')
    python = p.python; info.source = 'default';
end
if ~isempty(python) && ~exist(python,'file')
    warning('osld:noPython','%s (from %s) does not exist.',python,info.source);
    python = '';
end
if isempty(python), info.source = ''; end

%-Report
%--------------------------------------------------------------------------
if nargout == 0
    if isempty(python)
        fprintf('No Python with osl-dynamics found: install it with osld_install.\n');
    else
        v = osld_run('versions',struct(),struct('python',python));
        fprintf('Python %s: %s (from %s)\n',v.python,python,info.source);
        fprintf('osl-dynamics %s, TensorFlow %s, NumPy %s\n',v.osl_dynamics,v.tensorflow,v.numpy);
    end
    clear python
end
