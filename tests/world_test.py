import numpy as np

import gym_ctf
import gym_ctf.state.world as world
import gym_ctf.state.agent as agent
import gym_ctf.state.team as team
import gym_ctf.state.flag as flag

def test_default_flag_values():
    ts = get_teams()
    w = world.World(10,10, ts)

    assert w.flag_radius == 1
    assert w.flag_count == 3

def test_timestep():
    w = world.World(10,10,np.array([]))
    t = w.time
    w.timestep()
    assert t + 1 == w.time

def test_reset():
    w = world.World(10,10, get_teams())
    w.flags[0].take(1)
    w.flags[1].scoring_count = 2
    w.timestep()
    w.timestep()

    w.reset()

    assert not w.flags[0].taken
    assert not w.flags[1].taken
    assert w.flags[1].scoring_count == 0
    assert w.time == 0

def test_score_flags():
    a1 = agent.Agent((1,1),0,0)
    a2 = agent.Agent((1,2),0,0)
    ta = team.Team(np.array([a1, a2]), 0)
    
    b1 = agent.Agent((1,1),0,1)
    b2 = agent.Agent((4,4),0,1)
    tb = team.Team(np.array([b1, b2]), 1)
    
    f1 = flag.Flag((1,1), 1.5)
    f2 = flag.Flag((4,4), 0.5)

    f2.scoring_team = 0
    f2.scoring_count = 2

    w = world.World(10, 10, np.array([ta, tb]), np.array([f1, f2]), 1.5)

    w.score_flags()

    assert f1.scoring_team == 0
    assert f1.scoring_count == 1
    assert f2.scoring_team == 1
    assert f2.scoring_count == 1
def get_teams():
    a = agent.Agent()
    t0 = team.Team(np.array([a]))
    ts = np.array([t0])
    return ts
    
