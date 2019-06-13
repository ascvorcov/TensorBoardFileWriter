rem publish sample
call make_sample.bat

set out=.\bin\X64\Release\netcoreapp2.2\win-x64\publish\

rem copy native CNTK dependencies into bin folder
copy .\cntk.gpu.2.7.0\support\x64\Release\*.*  %out%
copy .\cntk.deps.mkl.2.7.0\support\x64\Dependency\*.* %out%
copy .\cntk.deps.cuda.2.7.0\support\x64\Dependency\*.* %out%
copy .\cntk.deps.cudnn.2.7.0\support\x64\Dependency\*.* %out%

copy TensorBoardFileWriter.dll %out%

