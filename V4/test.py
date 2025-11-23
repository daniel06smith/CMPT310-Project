import time
from racer import Racer

# Test headless
env = Racer(render_mode=None, track_num=1)
start = time.time()
for _ in range(10000):
    env.step(env.action_space.sample())
    if env.crashed:
        env.reset()
headless_time = time.time() - start
print(f"Headless: {10000/headless_time:.0f} steps/sec")

# Test human mode
env = Racer(render_mode="human", track_num=1)
start = time.time()
for _ in range(1000):  # fewer steps since it's slower
    env.step(env.action_space.sample())
    if env.crashed:
        env.reset()
human_time = time.time() - start
print(f"Human: {1000/human_time:.0f} steps/sec")