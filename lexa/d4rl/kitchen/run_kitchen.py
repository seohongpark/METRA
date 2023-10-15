import gym
import d4rl
import cv2
from envs.d4rl_envs import KitchenEnv
#from d4rl.kitchen.kitchen_envs import KitchenMicrowaveV0
import imageio

env = KitchenEnv()
env.reset()
done = False
imgs = []
for i in range(150):
    o, r, d, i = env.step(env.action_space.sample())
    print("microwave", i["microwave distance to goal"])
    im = env.render(mode="rgb_array")
    imgs.append(im)
    #cv2.imshow("env", im)
    #cv2.waitKey(1)
    #print(r)
#import ipdb ; ipdb.set_trace()
imageio.mimsave('out.gif', imgs)
