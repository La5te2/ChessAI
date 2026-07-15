Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = baseDir

cmd = "pythonw.exe src\board.py" & _
  " --gui 1" & _
  " --model models\candidate1.pth" & _
  " --device cpu" & _
  " --search-type only-mcts" & _
  " --mcts-sims 10" & _
  " --mcts-min-sims 0" & _
  " --mcts-batch-size 32" & _
  " --movetime-ms 0" & _
  " --c-puct 0.5" & _
  " --c-puct-base 19652" & _
  " --c-puct-factor 1.0" & _
  " --fpu-reduction 0.15" & _
  " --mcts-time-fraction 1.0" & _
  " --mate-plies 0" & _
  " --mate-topk 0" & _
  " --mate-nodes 0" & _
  " --mate-hash-mb 16" & _
  " --root-topn 8"

shell.Run cmd, 0, False
