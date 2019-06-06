# TensorBoardFileWriter
Port of python implementation of TensorBoardFileWriter for CNTK.

Contains implementation of TensorBoardFileWriter and sample C# .net core application which uses it.

CNTK python bindings contain implementation of TensorBoardProgressWriter, which dumps model learning
progress into log file format, accepted by TensorBoard visualization tool.

C# bindings are missing this implementation, although base ProgressWriter class is available.
Subclassing it doesn't get desired result - overloaded members are not being called, because
ProgressWriter itself is a thin wrapper around native ProgressWriter class pointer. SWIG tool
which is used to generate C# bindings actually support this type of callback to managed code from unmanaged part
(called 'directors'), and this is actually implemented for python. But for some reason, C# bindings
do not use this.

This repository attempts to work around this shortcoming by implementing library which connects 
to unmanaged pipeline used internally by CNTK, by unwrapping internal pointers and calling native code directly.

Code heavily relies on implementation details, and any library version change may break the solution.

*make_dll.bat*
Attempts to install CNTK.GPU nuget package (nuget expected to be available), 
and build main native dll from C++ code, using VC compiler (SDK is expected to be installed).

*make_sample.bat*
Builds and publishes .net core sample (LogisticRegression taken from CNTK samples), referencing native dll
and utilizing progress writer. Copies native libraries into bin folder

*run_sample.bat*
Runs built .net sample, which produces data in log folder for TensorBoard to read.
Starts TensorBoard service and opens browser.
