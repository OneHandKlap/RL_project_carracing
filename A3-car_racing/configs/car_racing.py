class config():
    # env config
    render_train     = False
    render_test      = False
    env_name         = "CarRacing-v0"
    overwrite_render = True
    record           = False # Throws error on Windows
    high             = 255.

    # output config
    output_path  = "results/car_racing_dqn/"
    model_output = output_path + "model.weights"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    #load_path         = "weights/model.weights"
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    #nsteps_train       = 2000000
    nsteps_train       = 5
    playing_time       = 1
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.00008
    lr_end             = 0.00005
    lr_nsteps          = 1 #500000
    eps_begin          = 0.5
    eps_end            = 0.1
    eps_nsteps         = 1 #1000000
    learning_start     = 1 #50000

    # discrete action space
    discrete_action_space = list({
    "brake": [0, 0, 1],
    "slight_turn_left": [-.3, 0, 0],
    "slight_turn_right": [.3, 0, 0],
    "slight_go": [0, .3, 0],
    "slight_go_left": [-.3, .3, 0],
    "slight_go_right": [.3, .3, 0],
    "slight_brake": [0, 0, .3],
    "slight_brake_left": [-.3, 0, .3],
    "slight_brake_right": [.3, 0, .3]
    }.values())