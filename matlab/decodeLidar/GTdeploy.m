username = 'target7';
% address = '192.168.42.32';
%address = '139.30.200.112';
address = '139.30.200.83';
%address = '10.0.10.32';
model = 'SimulinkModel';

target = GT.GenericTarget(username, address);

target.terminateAtTaskOverload = false;
target.terminateAtCPUOverload = false;
target.targetSoftwareDirectory = '~/GT/sendPoints/';
target.targetBitmaskCPUCores = '0x0000F000';
% 0000 0000 0000 0000 1111 0000 0000 0000.
% The 1 bits are at positions 12, 13, 14, and 15 (counting from 0 on the right):

%target.additionalCompilerFlags.DEBUG_MODE = true;
%target.portAppSocket = 65535;
% target.DownloadAllData; % to download all log data
%GT.DecodeDataFiles; % when to read log file
%#########################################################
%last deploy
%target.GenerateCode;
target.Deploy(model);

