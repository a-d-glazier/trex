# TODO: This doesnt work...
# def record_video(env, policy_exec, video_dir, num_episodes=1):
#     """
#     Record and save a video of the environment's execution.

#     Args:
#         env: The gymnasium environment.
#         policy_exec: The policy execution function.
#         video_dir: Directory to save the video.
#         num_episodes: Number of episodes to record.
#     """
#     # Ensure video_dir exists
#     if not os.path.exists(video_dir):
#         os.makedirs(video_dir)
    
#     # Wrap the environment with the RecordVideo wrapper
#     env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda episode_id: True)

#     for episode in range(num_episodes):
#         print(f"Starting episode {episode + 1}")
#         # Reset the environment and get the initial state and info
#         observations, info = env.reset(options={'start_loc': 0, 'goal_loc': 8})
        
#         done = False
#         for _ in range(100):
#             if done:
#                 break
#             action = policy_exec(observations)
#             print(f"Action taken: {action}")
#             observations, reward, done, truncated, info = env.step(action)
#             print(f"Reward: {reward}, Done: {done}")
        
#         print(f"Episode {episode + 1} finished.")
        
#     # Close the environment when done
#     env.close()