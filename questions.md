## March 6

#### How to pick start
When picking the `start` of the tree, in the $10^6$ Elad made the *arbitrary* choice of 0.56. 

We think, that this should scale with the Tidal Radius and with β.

Right now, it looks like
start = Rt/(200β).

Since there is the following condition for this
> Ensure that the regular grid cells are smaller than simulation cells

we can make a script that precalculates it for any given parameters.


#### Make a time script
Keeping .h5 files on local is dumb. We only need them during extraction. 

We should have a script that calculates the t/tfb for every snapshot and save it to a file. 

Run it once when you have all the files, then delete the .h5 files.

#### Same for box

#### Dream
one file,
1. downloads
2. extracts
3. makes box file and time file
4. deletes .h5 files