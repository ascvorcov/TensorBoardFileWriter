pushd .\bin\Release\netcoreapp2.2\win-x64\publish\
md log
sample.exe
start tensorboard --logdir=.\log --host:127.0.0.1
start chrome http://%COMPUTERNAME%:6006
popd