rem install CNTK.GPU nuget package, expects nuget to be available from command line.
nuget install CNTK.GPU -Version 2.7.0
rem set path to installed nuget package
set cntkpath=".\CNTK.GPU.2.7.0\build\native"
rem VS140 is expected to be installed, or at least SDK with compiler
call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" amd64
rem compile x64 native dll with exported api function and unmanaged tensor board file writer class
cl /EHsc /I "%cntkpath%\include" /LD /Tp TensorBoardFileWriter.cpp /link "%cntkpath%\lib\x64\Release\Cntk.Core-2.7.lib" /MACHINE:x64


