import gym
import numpy as np

def run_experiment(num_episodes, num_actions, epsilon,alpha,gamma=1):
    env = gym.make('CarRacing-v0')
    q_table=-1*np.ones(shape=(80,45,2,8))
    discrete_action_space={"turn_left":[-1,0,0],"turn_right":[1,0,0],"go":[0,1,0],"go_left":[-1,1,0],"go_right":[1,1,0],"brake":[0,0,1],"brake_left":[-1,0,1],"brake_right":[1,0,1]}
    discrete_action_list=list(discrete_action_space.values())
    
    def get_next_action(state,epsilon):
        if np.random.random()<epsilon:
            return np.argmax(q_table[state])
        else:
            return np.random.randint(3)

    def calc_TD(state,action):
        return reward + gamma*np.max(q_table[state])
    
    def update_q_table(state,action):
        q_table[state]+=alpha*calc_TD(x,y,action)

    def get_2d_slice(matrix, start_row, end_row, start_col, end_col):
        return np.array([row[start_col:end_col] for row in matrix[start_row:end_row]])
    
    def black_and_white(matrix):
        gray=(np.dot(matrix[...,:3],[.3,.6,.1]))
        return gray//126

    for episode in range(num_episodes):

        env.reset()
        for action_num in range(num_actions):
            env.render()

            if action_num == 0:
                #random action
                action=(discrete_action_list[np.random.randint(len(discrete_action_list))])
                observation,reward,done, info=env.step(action)

            else:
                #random action
                action=(discrete_action_list[np.random.randint(len(discrete_action_list))])
                observation,reward,done, info=env.step(action)

                #slice observation
                o=np.array(get_2d_slice(observation,0,80,25,70))
                #convert observation to grayscale
                gray_obs=black_and_white(o)
                print(gray_obs)
                #update Q table







            
    env.close()

if __name__=="__main__":
    run_experiment(2,100,90,.5)


