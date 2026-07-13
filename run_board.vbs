Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = baseDir

cmd = "pythonw.exe src\board.py" & _
  " --gui 1" & _
  " --model models\candidate.pth" & _
  " --device cpu" & _
  " --mcts-sims 100" & _
  " --mcts-min-sims 0" & _
  " --mcts-batch-size 32" & _
  " --movetime-ms 0" & _
  " --c-puct 0.8" & _
  " --c-puct-base 19652" & _
  " --c-puct-factor 1.0" & _
  " --fpu-reduction 0.15" & _
  " --mcts-time-fraction 1.0" & _
  " --mate-guard-plies 0" & _
  " --mate-guard-topk 0" & _
  " --mate-guard-nodes 0" & _
  " --root-topn 8"

shell.Run cmd, 0, False
