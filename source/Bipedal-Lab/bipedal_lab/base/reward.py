from isaaclab.utils import configclass


@configclass
class RewardManagerCfg:
    """Configuration class for :class:`~bipedal_lab.base.reward.RewardManager`
    """
    pass


"""

track linear
- world z 에 대한 xy-plane 에서의 velocity
- 느릴 때는 strict, 빠를 때는 loose 하게
- command difficulty에 따라 scale 변화

track angular
- world z 에 대해서만
- body vel vs foot vel

"""


class RewardManager:
    def __init__(self, cfg: RewardManagerCfg):
        pass

    def compute(self):
        pass


# pen lin
- for time interval (motion period)
- non periodical (for one step)

# pen ang
- for time interval (motion period)
- non periodical

# lin pos (height)
- for time interval (motion period)
- non periodical

# ang pos (orientation)
- for time interval (motion period)
- non periodical

# periodical reward term 에는 last period command 의 consistency 를 같이 고려 

# energy: E = int Fv dt

# impulse: J = int F dt

# stabilize: contact

# stabilize: joint limit

# stabilize: action delta^2

# stabilize: foot slip

# posture: default joint pos
- for time interval (motion period)

# posture: foot clearance

