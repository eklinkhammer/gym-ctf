import gym_ctf
import gym_ctf.state.flag as flag

def get_random_flag():
    return flag.Flag.random_flag(0, 0, 10, 10, 2)

def test_reset():
    f = get_random_flag()
    f.take(1)
    f.scoring_count = 10
    
    f.reset()

    assert not f.taken
    assert f.scoring_team is None
    assert f.scoring_count == 0

def test_take():
    f = get_random_flag()

    f.take(2)

    assert f.taken
    assert f.scoring_team == 2

def test_within_scoring_distance_1():
    f = flag.Flag((0,0), 5.1)
    within = (3,4)

    assert f.within_scoring_distance(within)

def test_within_scoring_distance_2():
    f = flag.Flag((0,0), 5.1)
    within = (-3,4)

    assert f.within_scoring_distance(within)

def test_within_scoring_distance_3():
    f = flag.Flag((0,0), 5.1)
    within = (-3,-4)

    assert f.within_scoring_distance(within)

def test_within_scoring_distance_4():
    f = flag.Flag((0,0), 5.1)
    too_far = (4,4)

    assert not f.within_scoring_distance(too_far)



