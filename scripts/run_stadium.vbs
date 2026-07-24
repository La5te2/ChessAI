Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

rootDir = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
shell.CurrentDirectory = rootDir

whiteMovetimeMs = 60000
blackMovetimeMs = 60000

whiteUci = "python.exe scripts\uci.py" & _
  " --arch gadus" & _
  " --model models\gadus\candidate1.pth" & _
  " --device cpu" & _
  " --search-type only-mcts" & _
  " --mcts-sims 3000" & _
  " --mcts-min-sims 0" & _
  " --mcts-batch-size 32" & _
  " --c-puct 1.0" & _
  " --c-puct-base 19652" & _
  " --c-puct-factor 1.0" & _
  " --fpu-reduction 0.15" & _
  " --repetition-policy-penalty 0.5" & _
  " --instant-mate-first"

blackUci = "python.exe scripts\uci.py" & _
  " --arch gadus" & _
  " --model models\gadus\current6.pth" & _
  " --device cpu" & _
  " --search-type only-mcts" & _
  " --mcts-sims 3000" & _
  " --mcts-min-sims 0" & _
  " --mcts-batch-size 32" & _
  " --c-puct 1.0" & _
  " --c-puct-base 19652" & _
  " --c-puct-factor 1.0" & _
  " --fpu-reduction 0.15" & _
  " --repetition-policy-penalty 0.5" & _
  " --instant-mate-first"

cmd = "pythonw.exe scripts\stadium.py" & _
  " --white-uci """ & whiteUci & """" & _
  " --black-uci """ & blackUci & """" & _
  " --white-movetime-ms " & whiteMovetimeMs & _
  " --white-multipv 5" & _
  " --black-movetime-ms " & blackMovetimeMs & _
  " --black-multipv 5" & _
  " --delay-ms 2000" & _
  " --max-plies 300"

shell.Run cmd, 0, False
