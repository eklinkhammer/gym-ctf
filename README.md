# gym-ctf
An openAI gym for capture-the-flag between multiagent teams. Currently, two modes are supported.

## Ctf
A capture the flag environment where two teams of agents attempt to capture flags by having more members of their team around the flag for the time required to capture.

## CtfSingleTeam
This capture the flag environment has one team using a hand-coded policy.

## Setting Constants
The following constants are fixed, but can be edited in source. Determining how to configure gym environments:

Number of agents on each team (vector, can vary per team)
Number of teams
Number of additional agents needed to score
Scoring distance
Time needed to score
Relationship between extra agents and time required