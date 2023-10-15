import akro

from garage.envs import EnvSpec

class AkroWrapperTrait:
    @property
    def spec(self):
        return EnvSpec(action_space=akro.from_gym(self.action_space),
                       observation_space=akro.from_gym(self.observation_space))

