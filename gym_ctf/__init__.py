from gym.envs.registration import register

register(
    id='ctf-v0',
    entry_point='gym_ctf.envs:CtfEnv'
)
register(
    id='ctf-singleteam-v0',
    entry_points='gym_ctf.envs:CtfSingleTeamEnv'
)
