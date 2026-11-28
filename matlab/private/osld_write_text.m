function osld_write_text(file,txt,mode)
% Write (or append to) a UTF-8 text file
%__________________________________________________________________________

if nargin < 3, mode = 'w'; end
d = fileparts(file);
if ~isempty(d) && ~exist(d,'dir'), mkdir(d); end
fid = fopen(file,mode,'n','UTF-8');
if fid < 0, error('osld:io','Cannot write %s.',file); end
fwrite(fid,txt,'char');
fclose(fid);
