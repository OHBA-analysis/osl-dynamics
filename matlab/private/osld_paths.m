function p = osld_paths(dir)
% Paths of an osl-dynamics installation made by osld_install
% FORMAT p = osld_paths(dir)
%
% dir      - installation folder: %LOCALAPPDATA%\osl-dynamics\matlab on
%            Windows, ~/.osl-dynamics/matlab elsewhere
%
% p        - .root, .settings (settings file), .default_dir, .dir, .env
%            (Python environment), .python (its executable), .bin, .uv,
%            .python_install (uv-managed Pythons), .cache, .bridge
%            (osld_bridge.py)
%__________________________________________________________________________

if ispc
    base = getenv('LOCALAPPDATA');
    if isempty(base), base = fullfile(getenv('USERPROFILE'),'AppData','Local'); end
    p.root = fullfile(base,'osl-dynamics');
else
    home = getenv('HOME');
    if isempty(home), error('osld:noHome','Cannot find the home folder (HOME).'); end
    p.root = fullfile(home,'.osl-dynamics');
end
p.settings    = fullfile(p.root,'matlab.json');
p.default_dir = fullfile(p.root,'matlab');
if nargin < 1 || isempty(dir), dir = p.default_dir; end
p.dir = char(dir);
p.env = fullfile(p.dir,'env');
if ispc
    p.python = fullfile(p.env,'Scripts','python.exe');
    p.uv     = fullfile(p.dir,'bin','uv.exe');
else
    p.python = fullfile(p.env,'bin','python');
    p.uv     = fullfile(p.dir,'bin','uv');
end
p.bin            = fullfile(p.dir,'bin');
p.python_install = fullfile(p.dir,'python');
p.cache          = fullfile(p.dir,'cache');
p.bridge         = fullfile(fileparts(fileparts(mfilename('fullpath'))),'python','osld_bridge.py');
