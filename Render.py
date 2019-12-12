from PIL import Image

def render(env, path, policy = None, num_episodes = 1):
    """Graphically renders specified number of episodes using the given policy and stores the 
    episode visualization as a gif.

    :param env:           Gym environment.
    :param path:          path where the gif generated will be stored.
    :param policy:        function which maps state to action. If None, the random policy is used.
    :param num_episodes:  number of episodes to render.
    """

    if policy is None:
        
        def policy(state):
            return env.action_space.sample()
        
    else:
        
        policy.custom_load(path)
        
    state = env.reset()
    frames = []
    ep_num = 0 
    
    frames.append(Image.fromarray(env.render(mode = 'rgb_array')))
    env.render()
        
    while ep_num < num_episodes:
        action = policy.select_clipped_action(state, exploration_noise = 0)
        state, _, done, _ = env.step(action)
        frames.append(Image.fromarray(env.render(mode = 'rgb_array')))
        env.render()

        if done:
            with open(path + 'test_' + str(ep_num) + '.gif', 'wb') as fp:
                im = Image.new('RGB', frames[0].size)
                im.save(fp, save_all = True, append_images = frames, duration = 100, loop = 1)
            ep_num += 1 
            
            state = env.reset()
            frames = []
            
            frames.append(Image.fromarray(env.render(mode = 'rgb_array')))
            env.render()