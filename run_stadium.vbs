Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

baseDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = baseDir

whiteUci = "python.exe src\uci_engine.py" & _
  " --model models\candidate0.pth" & _
  " --device cpu" & _
  " --search-type closed" & _
  " --mcts-sims 0"

blackUci = "python.exe src\uci_engine.py" & _
  " --model models\candidate0.pth" & _
  " --device cpu" & _
  " --search-type only-mcts" & _
  " --mcts-sims 1000" & _
  " --mcts-min-sims 1000" & _
  " --mcts-batch-size 32" & _
  " --c-puct 1.5" & _
  " --c-puct-base 19652" & _
  " --c-puct-factor 1.0" & _
  " --fpu-reduction 0.15"

cmd = "pythonw.exe src\stadium.py" & _
  " --white-uci """ & whiteUci & """" & _
  " --black-uci """ & blackUci & """" & _
  " --movetime-ms 10000" & _
  " --delay-ms 300" & _
  " --max-plies 240"

shell.Run cmd, 0, False
