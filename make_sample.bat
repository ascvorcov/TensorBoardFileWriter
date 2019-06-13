rem build native dll
call make_dll.bat

rem build and publish x64 version of sample app, for netcore 2.2
dotnet publish .\Sample.csproj --configuration Release --framework netcoreapp2.2 --runtime win-x64
