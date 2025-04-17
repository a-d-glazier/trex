#!/usr/bin/env python3

"""
Simple runner script for ICRL on LapGridWorld environment.
"""

import os
import argparse
from icrl_grid_world import icrl_grid_world

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ICRL on LapGridWorld")
    
    # Important parameters to configure from command line
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_iters", type=int, default=50, help="Number of ICRL iterations")
    parser.add_argument("--forward_timesteps", type=int, default=2000, 
                       help="Timesteps to collect for policy training in each iteration")
    parser.add_argument("--backward_iters", type=int, default=50, 
                       help="Number of iterations to train constraint net in each ICRL iteration")
    parser.add_argument("--expert_rollouts", type=int, default=10, 
                       help="Number of expert demonstrations to collect/use")
    parser.add_argument("--save_dir", type=str, default="results/icrl_grid_world", 
                       help="Directory to save results")
    parser.add_argument("--budget", type=float, default=0.0, 
                       help="Cost budget (0 = constrain all backward movements)")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to use (cpu or cuda)")
    parser.add_argument("--warmup_timesteps", type=int, default=1000, 
                       help="Warmup timesteps without cost")
    
    # Parse args and create complete config
    args = parser.parse_args()
    
    # Create all directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Fill in default parameters for ICRL
    config = argparse.Namespace(
        # Environment
        seed=args.seed,
        num_threads=1,
        max_episode_steps=200,
        expert_rollouts=args.expert_rollouts,
        expert_path=None,
        save_expert_path=os.path.join(args.save_dir, "expert_data.pkl"),
        
        # ICRL parameters
        n_iters=args.n_iters,
        warmup_timesteps=args.warmup_timesteps,
        forward_timesteps=args.forward_timesteps,
        backward_iters=args.backward_iters,
        cn_plot_every=5,
        reset_policy=False,
        
        # Normalization
        dont_normalize_obs=False,
        dont_normalize_reward=False,
        dont_normalize_cost=False,
        cn_normalize=False,
        clip_obs=20.0,
        
        # Constraint net
        cn_learning_rate=3e-4,
        anneal_clr_by_factor=1.0,
        cn_reg_coeff=0.0,
        cn_batch_size=64,
        cn_obs_select_dim=None,
        cn_acs_select_dim=None,
        cn_layers=[64, 64],
        no_importance_sampling=False,
        per_step_importance_sampling=False,
        cn_target_kl_old_new=10,
        cn_target_kl_new_old=10,
        train_gail_lambda=False,
        cn_eps=1e-5,
        
        # Policy
        policy_name="TwoCriticsMlpPolicy",
        policy_layers=[64, 64],
        vf_layers=[64, 64],
        cvf_layers=[64, 64],
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        reward_gamma=0.99,
        reward_gae_lambda=0.95,
        cost_gamma=0.99,
        cost_gae_lambda=0.95,
        clip_range=0.2,
        clip_range_reward_vf=None,
        clip_range_cost_vf=None,
        ent_coef=0.0,
        reward_vf_coef=0.5,
        cost_vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        penalty_initial_value=1.0,
        penalty_learning_rate=0.1,
        budget=args.budget,
        proportional_control_coeff=10.0,
        integral_control_coeff=0.0001,
        derivative_control_coeff=0.0,
        pid_delay=1,
        proportional_cost_ema_alpha=0.5,
        derivative_cost_ema_alpha=0.5,
        
        # Other
        save_dir=args.save_dir,
        cost_info_str="cost",
        verbose=1,
        device=args.device,
    )
    
    # Run ICRL
    print(f"Running ICRL on LapGridWorld with {args.n_iters} iterations")
    print(f"Saving results to {args.save_dir}")
    
    nominal_agent, constraint_net = icrl_grid_world(config)
    
    print(f"ICRL training complete! Models saved to {args.save_dir}") 