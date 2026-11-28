function s = osld_settings(s)
% Read the settings file, or merge fields into it
% FORMAT s = osld_settings        - read (empty struct if there is none)
% FORMAT s = osld_settings(s)     - set the fields of s and write the file
%__________________________________________________________________________

file = osld_paths().settings;
old  = struct();
if exist(file,'file')
    try
        old = jsondecode(osld_read_text(file));
    catch
        warning('osld:settings','Cannot read %s; it will be replaced.',file);
    end
end
if nargin < 1
    s = old;
    return
end
f = fieldnames(s);
for i = 1:numel(f), old.(f{i}) = s.(f{i}); end
s = old;
osld_write_text(file,jsonencode(s));
