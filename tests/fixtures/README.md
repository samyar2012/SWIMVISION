# Real-video regression fixture

`swimmer_start_dive.mp4` is a 181-frame excerpt (source frames 120–300)
from a race video supplied by the project owner. It covers the selected
swimmer standing on the rightmost block, diving, passing through splash, and
appearing in the water.

The selection box on fixture frame 0 is `(620, 150, 710, 340)` in original
848×464 pixels. The fixture exists specifically to prevent synthetic tracker
tests from accepting changes that lose or switch the swimmer in real footage.

`swimmer_camera_pan.mp4` is a 341-frame excerpt (source frames 2320–2660)
from the same upload. It covers a foreground spectator occlusion and a large
camera pan that previously caused the tracker to relink to spectators for six
seconds. Its frame-0 selection is `(278, 53, 407, 125)`.
