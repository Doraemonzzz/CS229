function [q, steps_per_episode] = qlearning(episodes)

% set up parameters and initialize q values
alpha = 0.05;
gamma = 0.99;
num_states = 100;
num_actions = 2;
actions = [-1, 1];
q = zeros(num_states, num_actions);
steps_per_episode = zeros(1, episodes);

for i=1:episodes,
  [x, s, absorb] =  mountain_car([0.0 -pi/6], 0);
  %%% YOUR CODE HERE
  %%% 找到第一步的动作对应的最大值和索引
  [maxq, a] = max(q(s, :));
  %%% 如果相同则随机
  if q(s, 1) == q(s, 2)
      a = ceil(rand * num_actions);
  end
  %%% 更新步数
  steps = 0;
  
  %%% 如果未吸收
  while(~absorb)
      %%% 找到下一步的位置
      [x, sn, absorb] =  mountain_car(x, actions(a));
      %%% 奖励
      reward = - double(~ absorb);
      %%% 找到动作对应的最大值和索引
      [maxq, an] = max(q(sn, :));
      %%% 如果相同则随机
      if q(s, 1) == q(s, 2)
        an = ceil(rand * num_actions);
      end
      %%% 找到最大的行动
      q(s, a) = (1 - alpha) * q(s, a) + alpha * (reward + gamma * maxq);
      %%% 更新状态
      a = an;
      s = sn;
      steps = steps + 1;
  end
  %%% 记录步数
  steps_per_episode(i) = steps;
end