function restore = osld_setenv(vars)
% Set environment variables until the returned object is cleared
% FORMAT restore = osld_setenv(vars)
%
% vars     - {n x 2} cell array of names and values; an empty value removes
%            the variable if it is set
%
% restore  - onCleanup object that puts back the previous values when
%            cleared (variables that were not set are removed)
%__________________________________________________________________________

n   = size(vars,1);
old = cell(n,1);
changed = false(n,1);
for i = 1:n
    old{i} = getenv(vars{i,1});
    if isempty(vars{i,2})
        if ~isempty(old{i}), unset(vars{i,1}); changed(i) = true; end
    else
        setenv(vars{i,1},vars{i,2}); changed(i) = true;
    end
end
restore = onCleanup(@() put_back(vars(changed,1),old(changed)));
end


function put_back(names,values)
for i = 1:numel(names)
    if isempty(values{i}), unset(names{i}); else, setenv(names{i},values{i}); end
end
end


function unset(name)
if exist('unsetenv') > 0 %#ok<EXIST> R2022b and later
    unsetenv(name);
else
    setenv(name,'');
end
end
