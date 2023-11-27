from gymnasium import spaces

from envs.batched_env import SequencedBatchedEnv, ParallelBatchedEnv 
from envs.normalize import VecNormalize
from envs.make_env import make_env
from envs.state import State
from envs.transition import Transition
from envs.evaluate import RecordScores, Evaluator


ATARI_ENVS = [
    "adventure",
    "airraid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bankheist",
    "battlezone",
    "beamrider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "choppercommand",
    "crazyclimber",
    "defender",
    "demonattack",
    "doubledunk",
    "elevator_action",
    "enduro",
    "fishingderby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "icehockey",
    "jamesbond",
    "journeyescape",
    "kangaroo",
    "krull",
    "kungfumaster",
    "montezumarevenge",
    "mspacman",
    "namethisgame",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "privateeye",
    "qbert",
    "riverraid",
    "roadrunner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "spaceinvaders",
    "stargunner",
    "tennis",
    "timepilot",
    "tutankham",
    "upndown",
    "venture",
    "videopinball",
    "wizardofwor",
    "yarsrevenge",
    "zaxxon",
]

def has_discrete_action_space(env):
    return isinstance(env.action_space, spaces.Discrete)


def is_atari_env(env_id):
    env_id = env_id.lower()
    for env in ATARI_ENVS:
        if env in env_id:
            return True
    return False