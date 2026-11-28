function txt = osld_read_text(file)
% Read a UTF-8 text file
%__________________________________________________________________________

fid = fopen(file,'r','n','UTF-8');
if fid < 0, error('osld:io','Cannot read %s.',file); end
txt = fread(fid,[1 Inf],'*char');
fclose(fid);
