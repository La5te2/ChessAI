# Chess Piece Geometry

`piece.inc` contains the fixed built-in geometry, while `pieces.gpack` contains
deduplicated and compressed indexed geometry for imported styles. Both are
generated from the SVG sources listed below. Imported SVG files are one-time
inputs and are not retained by the project. RhosGFX, Chessnut, and Spatial use
the Lichess repository at commit
`c54596c9e1569d378fe20f80d4c790c6fdee54c4`.

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

## Fantasy

- Author: Maurizio Monge
- Source: https://github.com/lichess-org/lila/tree/c54596c9e1569d378fe20f80d4c790c6fdee54c4/public/piece/fantasy
- License: MIT
- License text: https://opensource.org/license/mit

## Cburnett

- Author: Colin M.L. Burnett
- Source: https://commons.wikimedia.org/wiki/Template:SVG_chess_pieces
- License: Creative Commons Attribution-ShareAlike 3.0 Unported
- License text: https://creativecommons.org/licenses/by-sa/3.0/legalcode

The generated geometry is embedded in `Gadidae`. The application does not load
the original SVG files at runtime.
