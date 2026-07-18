Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = baseDir

cmd = "pythonw.exe src\simulator.py" & _
  " --model models\current5.pth" & _
  " --device cpu" & _
  " --search-type only-mcts" & _
  " --mcts-sims 3000" & _
  " --mcts-min-sims 0" & _
  " --mcts-batch-size 32" & _
  " --movetime-ms 0" & _
  " --c-puct 1.0" & _
  " --c-puct-base 19652" & _
  " --c-puct-factor 1.0" & _
  " --fpu-reduction 0.15" & _
  " --repetition-policy-penalty 0.5" & _
  " --instant-mate-first" & _
  " --progress-interval-ms 100" & _
  " --root-topn 8"

shell.Run cmd, 0, False
