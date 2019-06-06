rem build native dll
call make_dll.bat

rem build and publish x64 version of sample app, for netcore 2.2
dotnet publish .\Sample.csproj --configuration Release --framework netcoreapp2.2 --runtime win-x64
set out=.\bin\Release\netcoreapp2.2\win-x64\publish\

rem copy native CNTK dependencies into bin folder
copy .\cntk.gpu.2.7.0\support\x64\Release\*.*  %out%
copy .\cntk.deps.mkl.2.7.0\support\x64\Dependency\*.* %out%
copy .\cntk.deps.cuda.2.7.0\support\x64\Dependency\*.* %out%
copy .\cntk.deps.cudnn.2.7.0\support\x64\Dependency\*.* %out%

copy TensorBoardFileWriter.dll %out%

