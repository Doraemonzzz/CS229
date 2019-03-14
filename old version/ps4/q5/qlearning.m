function [q, steps_per_episode] = qlearning(episodes)

% set up parameters and initialize q values
alpha = 0.05;
gamma = 0.99;
num_states = 100;
num_actions = 2;
actions = [-1, 1];
q = zeros(num_states, num_actions);

for i=1:episodes,
  [x, s, absorb] =  mountain_car([0.0 -pi/6], 0);
  %%% YOUR CODE HERE
  
  
end