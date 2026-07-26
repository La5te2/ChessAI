# Chess Piece Geometry

`piece_meshes.inc` contains normalized triangle, gradient-color, and stroke-path
data generated from the following SVG chess sets in the Lichess repository at
commit `c54596c9e1569d378fe20f80d4c790c6fdee54c4`.

## RhosGFX

- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/rhosgfx
- License: CC0 1.0
- License text: https://creativecommons.org/publicdomain/zero/1.0/legalcode

## Chessnut

- Author: Alexis Luengas
- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/chessnut
- License: Apache License 2.0
- License text: https://www.apache.org/licenses/LICENSE-2.0

## Spatial

- Author: Maurizio Monge
- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/spatial
- License: MIT
- License text: https://opensource.org/license/mit

The generated geometry is compiled into `Gadidae`. The application does not
load the original SVG files at runtime.
