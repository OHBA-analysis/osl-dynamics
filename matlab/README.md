# osl-dynamics MATLAB interface: Windows test

This folder is a test version of a MATLAB interface to osl-dynamics.
`osld_windows_test` checks whether it works on your computer and writes a
report.

## Requirements

- Windows 10 or 11 on an x64 processor (Intel or AMD), or Windows 11 on ARM
  (e.g. Snapdragon) version 24H2 or later with the latest updates. To find
  out which you have, go to Settings > System > About ("System type").
- MATLAB R2023b or later; on Windows on ARM, R2026a or later.
- About 3 GB of free disk space and an internet connection.
- Administrator rights are not needed.

## Steps

1. Download
   https://github.com/OHBA-analysis/osl-dynamics/archive/refs/heads/matlab-windows-test.zip
   and unzip it anywhere.
2. In MATLAB, go to the `matlab` folder inside the unzipped folder and run

   ```matlab
   osld_windows_test
   ```

   This takes about 10–30 minutes. It installs Python, osl-dynamics and
   TensorFlow (about 1.5 GB) into `%LOCALAPPDATA%\osl-dynamics\matlab`, then
   trains a small model on simulated data.
3. Ctrl+C test: run

   ```matlab
   osld_windows_test('ctrlc')
   ```

   and press Ctrl+C in the Command Window when lines starting with `busy:`
   appear. Then run

   ```matlab
   osld_windows_test('processes')
   ```

4. Remove the installation:

   ```matlab
   osld_windows_test('uninstall')
   ```

5. Send `osld_windows_report.txt` and `osld_windows_log.txt` from the
   `matlab` folder. Your user and computer names are replaced by `<user>`
   and `<computer>` in both files. Please check them before sending.

## If the installation fails

- **Proxy:** if your network needs a proxy, run
  `osld_install(struct('proxy','http://proxy.example.ac.uk:8080'))` with
  your proxy's address, then run `osld_windows_test` again.
- **Programs blocked:** if the report says uv cannot be run, your
  organization may block programs in your user folders. Try another folder,
  e.g. `osld_install(struct('dir','C:\osl-dynamics'))`, then run
  `osld_windows_test` again.

Please send the report and log in either case.
